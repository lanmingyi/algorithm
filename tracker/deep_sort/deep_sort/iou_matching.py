# 加入绝对引用这个新特性。这样,你就可以用import string来引入系统的标准string.py,而用from pkg import string来引入当前目录下的string.py了.
from __future__ import absolute_import
import numpy as np
from . import linear_assignment


def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    # 预测框的左上，右下坐标
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    # 多个候选框的坐标
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    # 计算中间重合部分矩形框IOU大小
    # np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
    # np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
    # np.newaxis的功能:插入新维度,增加一个维度。对于[:,np.newaxis]和[np.newaxis,]是在np.newaxis这里增加1维。
    # 这样改变维度的作用往往是将一维的数据转变成一个矩阵，与代码后面的权重矩阵进行相乘， 否则单单的数据是不能这样相乘的。
    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    # 求面积
    area_intersection = wh.prod(axis=1)  # 返回给定轴上数组元素的乘积。axis=1指定计算每一行的乘积。
    area_bbox = bbox[2:].prod()  # axis=None, will calculate the product of all the elements in the input array
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)  # 返回IOU大小


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.  距离度量

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape  花费矩阵/成本矩阵     Profit Matrix利益矩阵
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    # 创建ID
    if track_indices is None:
        track_indices = np.arange(len(tracks))  # 函数返回一个包含起点不包含终点的固定步长的排列
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))  # 返回一个给定形状和类型的用0填充的数组；
    # 枚举，对于一个可迭代的（iterable）/可遍历的对象（如列表、元组、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
    for row, track_idx in enumerate(track_indices):  # row：索引，track_idx：值
        # if语句用于排除age>1的轨迹，因为传入的值是track的所有值，大部分已经进行过处理了，只需要处理age=1的。如果注释掉的话，可能只是运行时间略微增加，对结果影响不大，因为与之前重复。
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        # 首先计算这些框两两之间的iou，经由1-iou得到cost_matrix
        bbox = tracks[track_idx].to_tlwh()  # 计算IOU的cost值，cost越小，IOU越大
        # b=np.array(a)与c=np.asarray(a)，将输入转为矩阵格式，当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值。 当输入为数组时，改变a，b不会变，c会变
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix
