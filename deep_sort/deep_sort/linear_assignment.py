from __future__ import absolute_import
import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
from . import kalman_filter
# import kalman_filter


INFTY_COST = 1e+5


def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.  解决线性分配问题

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.  控制阈值，与成本大于此值的关联被忽略
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:  # 创建ID
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    # 下面这个distance_metric包括外观（深度特征）和运动信息（马氏距离）
    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    # 将经过马氏距离处理的矩阵cost_matrix继续经由max_distance处理
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5  # 程序中把cost大于阈值0.7的，都设置成了0.7
    # 这样把cost_matrix作为匈牙利算法的输入，得到线性匹配结果
    indices = linear_assignment(cost_matrix)  # 线性排列（匈牙利算法），筛选出匹配的detection和track
    # np.hastack：行连接，在水平方向上平铺。np.vstack：列连接，在竖直方向上堆叠
    indices = np.hstack([indices[0].reshape(((indices[0].shape[0]), 1)), indices[1].reshape(((indices[0].shape[0]), 1))])
    # r_ind, c_ind = linear_assignment(cost_matrix)
    # indices = np.vstack((r_ind, c_ind)).T

    # 对匹配结果进行筛选，删去两者差距太大的
    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:  # 如果检测的ID不在所得indices（索引）中，则将其添加到未匹配检测序列中
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        # 如果某个组合的cost值大于阈值，这样的组合还是认为是不match的，相应的，还要把组合中的检测框和跟踪框都踢到各自的umatched列表中
        if cost_matrix[row, col] > max_distance:  # 再次筛选，大于阈值，列为未匹配序列，小于阈值，进入匹配序列
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):
    """Run matching cascade.  级联匹配。
    因为这个匹配操作需要从刚刚匹配成功的trk循环遍历到最多已经有30次（cascade_depth）没有匹配的trk

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))  # list方法将元组转换为列表
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    # 每次循环收集未匹配的tracker ID，并将这些ID进行再次匹配。具体算法思路请参考论文
    # 优先匹配消失最短的object
    for level in range(cascade_depth):
        # print("level: ")
        # print(level)
        if len(unmatched_detections) == 0:  # No detections left
            break  # 跳出循环不再执行

        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        # print("track_indices_l: ")
        # print(track_indices_l)
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue  # 跳出本次循环，执行下一次

        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections)
        matches += matches_l  # 组合各层的匹配
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))  # set创建一个无序不重复元素集
    return matches, unmatched_tracks, unmatched_detections


# 门控矩阵的作用就是通过计算卡尔曼滤波的状态分布和测量值之间的距离对代价矩阵进行限制。
# 代价矩阵中的距离是Track和Detection之间的表观相似度，假如一个轨迹要去匹配两个表观特征非常相似的Detection，这样就很容易出错，
# 但是这个时候分别让两个Detection计算与这个轨迹的马氏距离，并使用一个阈值gating_threshold进行限制，所以就可以将马氏距离较远的那个Detection区分开，可以降低错误的匹配。
def gate_cost_matrix(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    # 进行运动信息约束。  根据通过卡尔曼滤波获得的状态分布，使成本矩阵中的不可行条目无效。
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]  # 9.4877
    # 先将各检测结果由[x,y,w,h]转化为[center x, center y, aspect ration, height]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        # 每个tracker与detection的马氏距离
        # 对每个trk，计算其预测结果和检测结果之间的马氏距离，
        # 并将在cost_matrix中，相应的trk的马氏距离大于阈值（gating_threshold）的值设为100000（gated_cost，相当于设为无穷大）
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost  # 设置为inf
    return cost_matrix
