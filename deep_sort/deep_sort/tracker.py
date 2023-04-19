from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


# Tracker类是最核心的类，Tracker中保存了所有的轨迹信息，负责初始化第一帧的轨迹、卡尔曼滤波的预测和更新、负责级联匹配、IOU匹配等等核心工作。
# 负责调用卡尔曼滤波来预测track的新状态+进行匹配工作+初始化第一帧
# Tracker调用update或predict的时候，其中的每个track也会各自调用自己的update或predict
class Tracker:
    """
    This is the multi-target tracker.  是一个多目标tracker，保存了很多个track轨迹，

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):  # 调用的时候，后边的参数全部是默认的
        self.metric = metric  # metric是一个类，用于计算距离(余弦距离或马氏距离)
        self.max_iou_distance = max_iou_distance  # 最大iou，iou匹配的时候使用。程序中把cost大于阈值0.7的，都设置成了0.7
        self.max_age = max_age  # 直接指定级联匹配的cascade_depth参数
        self.n_init = n_init  # n_init代表需要n_init次数的update才会将track状态设置为confirmed

        self.kf = kalman_filter.KalmanFilter()  # 卡尔曼滤波器
        self.tracks = []  # 保存一系列轨迹
        self._next_id = 1  # 下一个分配的轨迹id

    # 遍历每个track都进行一次预测
    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    # 进行测量的更新和轨迹管理
    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        # 首先进行检测结果和跟踪预测结果的匹配。假设第一帧有18个检测框，第二帧有12个检测框，利用卡尔曼滤波第二帧会产生18个跟踪预测框
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set. 针对匹配上（match）的，要用检测结果去更新相应的tracker参数。更新内容在track.py中
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:  # 针对未匹配的tracker,调用mark_missed标记。track失配，若待定则删除，若update时间很久也删除
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:  # 针对unmatched_detections，要为其创建新的tracker
            self._initiate_track(detections[detection_idx])  # 针对未匹配的detection，detection失配，进行初始化
        self.tracks = [t for t in self.tracks if not t.is_deleted()]  # 得到最新的tracks列表，保存的是标记为confirmed和Tentative的track

        # Update distance metric. 更新已经确认的trk的特征集
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]  # 获取所有confirmed状态的track_id
        # print('************', active_targets)
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features  # 将tracks列表拼接到features列表
            targets += [track.track_id for _ in track.features]  # 获取每个feature对应的track_id
            track.features = []
        # print(features)
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)  # 距离度量中的特征集更新

    # 主要功能是进行匹配，找到匹配的，未匹配的部分
    # 功能： 用于计算track和detection之间的距离，代价函数。需要使用在KM算法之前
    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            # 计算当前帧每个新检测结果的深度特征与这一层中每个trk已保存的特征集之间的余弦距离矩阵，下面三行完成
            # 具体过程是针对trk的每个特征（因为每个trk都有一个特征集），计算它们与当前这14个det的特征之间的（1-余弦距离）。然后取最小值作为trk与检测结果之间的计算值
            features = np.array([dets[i].feature for i in detection_indices])
            # print(features.shape)
            targets = np.array([tracks[i].track_id for i in track_indices])
            # print(targets.shape)
            cost_matrix = self.metric.distance(features, targets)  # 通过最近邻计算出代价矩阵 cosine distance
            # print("cost_matrix: ")
            # print(cost_matrix)
            # 在cost_matrix中，进行运动信息约束
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)   # 计算马氏距离,得到新的状态矩阵
            # print("cost_matrix: ")
            # print(cost_matrix)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks. 将已存在的tracker分为confirmed_tracks和unconfirmed_tracks
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features. 进行级联匹配，得到匹配的track、不匹配的track、不匹配的detection
        # 针对之前已经confirmed_tracks,将它们与当前的检测结果进行级联匹配。  仅仅对确定态的轨迹进行级联匹配
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)  # gated_metric->cosine distance

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        # 进行IoU的匹配。将所有状态为未确定态的轨迹（unconfirmed_tracks）和刚刚没有匹配上的轨迹unmatched_tracks_a组合为iou_track_candidates，
        # iou_track_candidates与还没有匹配上的检测结果unmatched_detections进行IOU匹配
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]  # 刚刚没有匹配上

        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]  # 已经很久没有匹配上

        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b  # 组合两部分match得到的结果
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections  # 经过上述处理后，依据IOU得到当前的匹配结果

    def _initiate_track(self, detection):
        # 根据初始检测位置初始化新的kalman滤波器的mean和covariance
        mean, covariance = self.kf.initiate(detection.to_xyah())
        # 初始化一个新的tracker
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1  # _next_id+1理解：多一个tracker，id也多一个
