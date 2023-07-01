from __future__ import print_function

import os
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter  # 卡尔曼滤波

np.random.seed(0)


# 算法和程序都比较简单。程序依赖 scikit-learn 所提供的 linear_assignment 实现匈牙利匹配。KalmanFilter 由 FilterPy 提供。
def linear_assignment(cost_matrix):  # 匈牙利匹配
    try:
        import lap  # Linear Assignment Problem solver
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area，比例就是面积
    r = w / float(h)  # 纵横比，横纵比
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right，右下
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.这个类表示被观察为bbox的单个跟踪对象的内部状态
    # 使用初始边界框初始化跟踪器
    # 如果不关心具体方式，知道这里定义了一种恒速模型就好了
    什么是卡尔曼滤波？你可以在任何含有不确定信息的动态系统中使用卡尔曼滤波，对系统下一步的走向做出有根据的预测，即使伴随着各种干扰，卡尔曼滤波总是能指出真实发生的情况
    """
    # 此类表示观测目标框所对应跟踪对象的内部状态。定义等速模型
    # 内部使用KalmanFilter，7个状态变量，4个观测输入
    # F是状态变换模型，H是观测函数，R为测量噪声矩阵，P为协方差矩阵，Q为过程噪声矩阵
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox. 用观察到的bbox更新状态向量
        """
        # 使用观察到的目标框更新状态向量，filterpy.kalman.KalmanFilter.update会根据观测修改内部状态估计self.kf.x
        # 重置self.time_since_update，清空self.history。
        self.time_since_update = 0  # 只有在update的时候，time_since_update才会重置为0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # 推进状态向量并返回预测的边界框估计。
        # 将预测结果追加到self.history。由于get_state直接访问self.kf.x，所以self.history没有用到。
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.返回当前边界框估计值。
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    将物体检测的BBox与卡尔曼滤波器预测的跟踪BBox匹配
    Assigns detections to tracked object (both represented as bounding boxes)，将检测分配给被跟踪的对象（都表示为边界框）
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers，返回三个列表：匹配，不匹配的检测和不匹配的跟踪器
    """
    # 这里命名不准确，应该是将检测框关联到跟踪目标（objects）或者轨迹（tracks），而不是跟踪器（trackers）。
    # 跟踪器数量为0则直接构造结果。
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # 利用匈牙利算法线性分配得到match组合
            matched_indices = linear_assignment(-iou_matrix)  # IOU不支持数组计算，逐个计算两两间的交并比，调用linear_assignment进行匹配
    else:
        matched_indices = np.empty(shape=(0, 2))

    # 记录未匹配的检测框及轨迹
    unmatched_detections = []
    for d, det in enumerate(detections):  # 没有匹配上的物体检测BBox放入unmatched_detections列表，表示有新的物体进入画面，后面要新增跟踪器追踪新物体
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):  # 没有匹配上的卡尔曼滤波器预测的BBox放入unmatched_trackers列表，表示之前跟踪的物体离开画面了，后面要删除对应的跟踪器
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU，过滤掉IOU低的匹配
    matches = []
    # 遍历matched_indices矩阵，将IOU值小于iou_threshold的匹配结果分别放入unmatched_detections与unmatched_trackers列表中
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))  # 匹配上的卡尔曼滤波器预测的BBox与物体检测BBox以[[d,t]...]的形式放入matches矩阵
    # 初始化用列表，返回值用Numpy.array
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)  # 返回跟踪成功的物体矩阵，新增物体的矩阵，离开画面的物体矩阵


class Sort(object):
    """
    SORT算法包装类，update方法实现了SORT算法，输入是当前帧中所有物体的检测BBox集合，包括物体的score，
    输出是当前帧的物体跟踪BBox集合，包括物体跟踪的ID
    """

    # Sort 是一个多目标跟踪器，管理多个KalmanBoxTracker对象。
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        # 根据当前所有的卡尔曼跟踪器的个数（等于上一帧中被跟踪的物体个数）创建二维矩阵trks，行号为卡尔曼跟踪器标识，列向量为跟踪BBox与物体跟踪ID
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):  # 循环遍历卡尔曼跟踪器列表
            pos = self.trackers[t].predict()[0]  # 用卡尔曼跟踪器t产生对应物体在当前帧中预测的BBox
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]  # trk存储跟踪器的预测
            if np.any(np.isnan(pos)):
                to_del.append(t)
        # numpy.ma.masked_invalid屏蔽出现无效值的数组（NaN 或 inf）
        # numpy.ma.compress_rows压缩包含掩码值的2-D数组的整行。这相当于np.ma.compress_rowcols(a, 0)，有关详细信息，请参阅extras.compress_rowcols。
        # reversed返回反向iterator.seq必须是具有__reversed__()方法的对象，或者支持序列协议（__len__()方法和__getitem__()方法，整数参数从0开始）。
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  # trks中存放了上一帧中被跟踪的所有物体在当前帧中预测的BBox
        for t in reversed(to_del):
            self.trackers.pop(t)  # 逆向删除异常的跟踪器，防止破坏索引。压缩能够保证在数组中的位置不变。
        # 将物体检测的BBox与卡尔曼滤波器预测的跟踪BBox匹配，获得跟踪成功的物体矩阵，新增物体的矩阵，离开画面的物体矩阵
        # 判断dets和trks的match情况
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        # 跟踪成功的物体BBox信息更新到对应的卡尔曼滤波器
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        # 初始化创建卡尔曼跟踪器。  新增的物体要创建新的卡尔曼滤波器对象用于跟踪。  为无法匹配的检测创建和初始化新的跟踪器
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # print("trk.id = %d" %trk.id)
                # print(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
                # 跟踪成功的物体BBox与Id放入ret列表中
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet  删除死掉的跟踪序列
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)  # 离开画面/跟踪失败的物体从物体的卡尔曼跟踪器跟踪列表中删除
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))  # 返回当前画面中所有被跟踪物体的BBox与ID二维矩阵[[x1,y1,x2,y2,id1]...[...]]，在main函数


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # all train
    # sequences = ['PETS09-S2L1', 'TUD-campus', ]  # Motchallenge的数据集
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0  # 总时间
    total_frames = 0  # 总帧数
    colours = np.random.rand(32, 3)  # used only for display，Bbox颜色种类
    if display:  # 如果要显示追踪结果到屏幕上，2D绘图初始化
        if not os.path.exists('mot_benchmark'):
            print('\n\tERROR: mot_benchmark link not found!\n\n Create a symbolic link to the MOT benchmark\n '
                  '(https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n   '
                  '$ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    if not os.path.exists('output'):
        os.makedirs('output')
    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    # for seq in sequences:  # 循环依次处理多个数据集
    for seq_dets_fn in glob.glob(pattern):  # 循环依次处理多个数据集
        # 创建跟踪器对象，用来计算被跟踪物体的下一帧BBox
        mot_tracker = Sort(max_age=args.max_age,
                           min_hits=args.min_hits,
                           iou_threshold=args.iou_threshold)  # create instance of the SORT tracker
        # det.txt：指定的数据集中被检测物体的BBox列表第0列代表帧数，第2-6列代表物体的BBox,读取该det.txt到矩阵seq_dets
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')  # load detections
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]

        with open(os.path.join('output', '%s.txt' % seq), 'w') as out_file:
            print("Processing %s." % seq)
            # seq_dets第1列最大的值决定了本数据集的总帧数，循环逐帧处理
            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]  # 从seq_dets矩阵中取出第1列等于当前帧的所有检测物体的BBox到矩阵dets
                dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                if display:  # 如果要显示跟踪结果，那么先将数据集中当前帧的jpg文件显示到屏幕上
                    fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg' % frame)
                    im = io.imread(fn)  # skimage.io.imread 从文件加载图像
                    ax1.imshow(im)
                    plt.title(seq + ' Tracked Targets')

                start_time = time.time()  # 获得当前的时间
                trackers = mot_tracker.update(dets)  # 将当前帧中所有检测物体的BBox送入到SORT算法，获得对所有物体的跟踪计算结果BBox
                cycle_time = time.time() - start_time  # 获得SORT算法的耗时
                total_time += cycle_time  # 总时间累计

                for d in trackers:  # 再将SORT更新的所有跟踪结果逐个画到当前帧并显示到屏幕上
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                          file=out_file)
                    if display:
                        d = d.astype(np.int32)
                        # matplotlib.axes.Axes.add_patch 将补丁p添加到轴补丁列表中；剪辑框将设置为Axes剪切框。如果未设置变换，则将其设置为transData。返回补丁。
                        ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,
                                                        ec=colours[d[4] % 32, :]))
                if display:
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
        total_time, total_frames, total_frames / total_time))

    if display:
        print("Note: to get real runtime results run without the option: --display")
