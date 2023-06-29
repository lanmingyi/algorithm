# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from lib.utils.cython_bbox import bbox_overlaps

from lib.config import config as cfg
from lib.utils.bbox_transform import bbox_transform

'''
anchor_target_layer主要针对RPN的输出进行处理，
对RPN的输出结果加工，对anchor打上标签，
然后通过与Gt的比对，计算出与真实框的偏差
'''


def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
    """Same as the anchor target layer in original Fast/er RCNN """
    A = num_anchors  # 我的理解：单通道anchor的数量
    total_anchors = all_anchors.shape[0]  # 我的理解：多通道anchor的数量，也就是说所有的框
    K = total_anchors / num_anchors
    im_info = im_info[0]

    # allow boxes to sit over the edge by a small amount
    # _allowed_border代表框是否允许贴近image的边缘，0代表不允许
    _allowed_border = 0

    # map of shape (..., H, W)
    # 输出rpn_cls_score的height和width
    height, width = rpn_cls_score.shape[1:3]

    # only keep anchors inside the image
    # 只保留在img范围内的box，过滤掉越界的
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &  # 假如_allowed_border=100，说明x>=-100，也就是允许x为负，也就是说明x在边界外面了，0自然就不允许
        (all_anchors[:, 1] >= -_allowed_border) &  # 同理
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0]  # 这样看all_anchors的存储方式是[x,y,w,h]

    # keep only inside anchors
    # 只保留限定的anchors，例如_allowed_border=0，只保留在图像内的框
    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    # 只给不越界的anchor赋值，首先全部标记负样本
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    # 计算anchors和gt_boxes的重合率
    # np.ascontiguousarray返回一个指定数据类型的连续数组，转存为顺序结构的数据
    # anchors*gt_boxes
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1)  # 求每一行的最大值下标，应该就是每个anchor对应分数最大的gt_box
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]  # 将每行得到的最大值保存
    gt_argmax_overlaps = overlaps.argmax(axis=0)  # 求每一列的最大值下标，应该就是每个gt_box对应分数最大的anchor
    gt_max_overlaps = overlaps[gt_argmax_overlaps,  # 同理，保存每列的最大值
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]  # 保留那些相等的索引

    # RPN_CLOBBER_POSITIVES=True 代表 如果同时满足正负条件设置为负样本
    # if RPN_CLOBBER_POSITIVES = False 将所有满足负样本label阙值的anchor标记为0
    if not cfg.FLAGS.rpn_clobber_positives:
        # assign bg labels first so that positive labels can clobber them
        # first set the negatives
        labels[max_overlaps < cfg.FLAGS.rpn_negative_overlap] = 0

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1   # 前景标签，对于gt_boxes,和anchors重合率最大的检测框标记为1

    # fg label: above threshold IOU
    labels[max_overlaps >= cfg.FLAGS.rpn_positive_overlap] = 1  # 将anchor最大的分并且大于RPN_POSITIVE_OVERLAP的标记为1

    # 对那些与某个框的最大交叠值的阈值小于负样本的，设置为0，也就是负样本
    # 意思是，有些检测结果是某些框的最大交叠结果，但是交叠还是低于负样本阈值了，这些样本作为负样本
    if cfg.FLAGS.rpn_clobber_positives:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.FLAGS.rpn_negative_overlap] = 0

    # subsample positive labels if we have too many
    # 如果还有过多的正样本，再采样一次，平衡正负样本
    '''
    # Max number of foreground examples
    __C.TRAIN.RPN_FG_FRACTION = 0.5 # 代表每次训练RPN的比例
    '''
    num_fg = int(cfg.FLAGS.rpn_fg_fraction * cfg.FLAGS.rpn_batchsize)
    fg_inds = np.where(labels == 1)[0]  # 找出所有正样本的下标,1*len(fg_inds)
    if len(fg_inds) > num_fg:  # 如果超过了制定的数量，使用random.choice()随机采样
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1  # 将随机出来额外的标记为负样本

    # subsample negative labels if we have too many
    # 同理，但对num的设置有所不同
    num_bg = cfg.FLAGS.rpn_batchsize - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    # argmax_overlaps:每个anchor对应的最大gt_box下标
    # gt_boxes[argmax_overlaps, :] : 满足每个anchor对应的gt_box下标的gt_box信息
    # anchors:[x,y,w,h]
    # 计算与anchor与最大重叠的GT的偏移量
    # return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    # 设置正样本回归 loss 的权重
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # only the positive ones have regression targets
    # 对正样本赋初值
    bbox_inside_weights[labels == 1, :] = np.array(cfg.FLAGS2["bbox_inside_weights"])

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.FLAGS.rpn_positive_weight < 0:  # __C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0 设置-1使用统一的权重
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)  # labels>=0的个数，也就是样本数目
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((cfg.FLAGS.rpn_positive_weight > 0) &
                (cfg.FLAGS.rpn_positive_weight < 1))
        # 如果不是-1，__C.TRAIN.RPN_POSITIVE_WEIGHT = p(0<p<1)，positive权重就是p/{num positive},negative权重为(1-p)/{num negative}
        positive_weights = (cfg.FLAGS.rpn_positive_weight /
                            np.sum(labels == 1))
        negative_weights = ((1.0 - cfg.FLAGS.rpn_positive_weight) /
                            np.sum(labels == 0))
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    # map up to original set of anchors
    # _unmap的作用是从在框内的anchor又扩展回全部的anchor尺寸，fill代表在边界外的anchor需要填充的数字
    # 也就是映射回原来的total_anchor集合
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    # labels
    # A = num_anchors  # 我的理解：单通道anchor的数量
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4))

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_outside_weights = bbox_outside_weights
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    # 在刚开始处理时有可能会去掉一些边界外的，这些数据就一直未处理，还原原来的大小，对未处理的填充fill值
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)  # 生成一些随机数
        ret.fill(fill)  # 然后全部填充fill
        ret[inds] = data  # 再把原来的数据给赋值回去
    else:  # 多维的
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5
    # 返回的是(targets_dx, targets_dy, targets_dw, targets_dh))
    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
