# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope

from lib.config import config as cfg
from lib.layer_utils.anchor_target_layer import anchor_target_layer
from lib.layer_utils.proposal_layer import proposal_layer
from lib.layer_utils.proposal_target_layer import proposal_target_layer
from lib.layer_utils.proposal_top_layer import proposal_top_layer
from lib.layer_utils.snippets import generate_anchors_pre


class Network(object):
    def __init__(self, batch_size=1):
        self._feat_stride = [16, ]
        self._feat_compress = [1. / 16., ]
        self._batch_size = batch_size
        self._predictions = {}  # 保存预测结果
        self._losses = {}  # 保存损失值
        self._anchor_targets = {}  # 保存预设anchor的坐标
        self._proposal_targets = {}
        self._layers = {}  # 保存网络
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self._variables_to_fix = {}  # 保存fine-tune时需要固定值的变量

    # Summaries #
    def _add_image_summary(self, image, boxes):
        # add back mean
        image += cfg.FLAGS2["pixel_means"]
        # bgr to rgb (opencv uses bgr)
        channels = tf.unstack(image, axis=-1)
        image = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
        # dims for normalization
        width = tf.to_float(tf.shape(image)[2])
        height = tf.to_float(tf.shape(image)[1])
        # from [x1, y1, x2, y2, cls] to normalized [y1, x1, y1, x1]
        cols = tf.unstack(boxes, axis=1)
        boxes = tf.stack([cols[1] / height,
                          cols[0] / width,
                          cols[3] / height,
                          cols[2] / width], axis=1)
        # add batch dimension (assume batch_size==1)
        #assert image.get_shape()[0] == 1
        boxes = tf.expand_dims(boxes, dim=0)
        image = tf.image.draw_bounding_boxes(image, boxes)

        return tf.summary.image('ground_truth', image)

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        # 作用是将输入的Tensor中0元素在所有元素中所占的比例计算并返回
        # 因为relu激活函数有时会大面积的将输入参数设为0，所以此函数可以有效衡量relu激活函数的有效性。
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    # Custom Layers #
    def _reshape_layer(self, bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name):
            # change the channel to the caffe format
            # 输入bottom为[b,h,w,2A]
            # to_caffe为[b,2A,h,w]
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # then force it to have channel 2
            # reshaped为[1,2,b*A*h,w]，且[0,0,b*A*h,w]对应bg，[0,1,b*A*h,w]对应fg
            reshaped = tf.reshape(to_caffe, tf.concat(axis=0, values=[[self._batch_size], [num_dim, -1], [input_shape[2]]]))
            # then swap the channel back
            # to_tf为[1,b*A*h,w,2]
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    def _softmax_layer(self, bottom, name):
        if name == 'rpn_cls_prob_reshape':
            # 以上面to_tf为例，to_tf为[b,A*h,w,2]
            input_shape = tf.shape(bottom)
            # bottom_reshaped为[b*A*h*w,2] 所有像素点的9个anchor全部平铺成列，平铺时遍历的顺序依次是w,h,A,b
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)

    # 对rpn计算结果roiproposals的优选
    # 当TEST.MODE = 'top'使用proposal_top_layer
    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        '''
        对rpn计算结果roi proposals的优选
        当TEST.MODE = 'top'使用proposal_top_layer
        当TEST.MODE = 'nms'使用proposal_layer
        默认使用nms，作者说top模式效果更好，但速度慢
        '''
        with tf.variable_scope(name):
            rois, rpn_scores = tf.py_func(proposal_top_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                           self._feat_stride, self._anchors, self._num_anchors],
                                          [tf.float32, tf.float32])
            rois.set_shape([cfg.FLAGS.rpn_top_n, 5])
            rpn_scores.set_shape([cfg.FLAGS.rpn_top_n, 1])

        return rois, rpn_scores

    # 当TEST.MODE = 'nms'使用proposal_layer，
    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name):
            rois, rpn_scores = tf.py_func(proposal_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                           self._feat_stride, self._anchors, self._num_anchors],
                                          [tf.float32, tf.float32])
            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])

        return rois, rpn_scores

    # roi_pooling
    def _crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name):
            # 提取batch_id,并转为一行，为全0，目的是指定所有crop均来自同一张图片（输入本来也就只有一张特征图）
            # rois为[batch_ids,xmin,ymin,xmax,ymax]
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bboxes
            bottom_shape = tf.shape(bottom)
            # self._feat_stride[0]=16 经过4次pooling
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be backpropagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            # 为了进行max_pooling，将范围扩大到14*14，这样经过下面的max_pooling得出的结果就是7*7
            pre_pool_size = cfg.FLAGS.roi_pooling_size * 2
            # crops为[num_boxes, crop_height, crop_width, depth]，一个bottom会输出num_boxes个图像
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME')

    # dropout，概率为ratio
    def _dropout_layer(self, bottom, name, ratio=0.5):
        return tf.nn.dropout(bottom, ratio, name=name)

    # 根据gt给所有预设anchor计算标签和偏移量
    def _anchor_target_layer(self, rpn_cls_score, name):
        with tf.variable_scope(name):
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer,
                [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
                [tf.float32, tf.float32, tf.float32, tf.float32])

            rpn_labels.set_shape([1, 1, None, None])
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            self._anchor_targets['rpn_labels'] = rpn_labels  # rpn_labels=[1, 1, A * height, width] 整张图所有预设anchor的标签
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets  # rpn_bbox_targets=[1, height, width, A * 4] 整张图所有anchor的偏移量
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights  # rpn_bbox_inside_weights=[1, height, width, A * 4] 在图片范围内的边框权重
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

            self._score_summaries.update(self._anchor_targets)

        return rpn_labels

    def _proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name):
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer,
                [rois, roi_scores, self._gt_boxes, self._num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

            rois.set_shape([cfg.FLAGS.batch_size, 5])
            roi_scores.set_shape([cfg.FLAGS.batch_size])
            labels.set_shape([cfg.FLAGS.batch_size, 1])
            bbox_targets.set_shape([cfg.FLAGS.batch_size, self._num_classes * 4])
            bbox_inside_weights.set_shape([cfg.FLAGS.batch_size, self._num_classes * 4])
            bbox_outside_weights.set_shape([cfg.FLAGS.batch_size, self._num_classes * 4])
            '''
            rpn_rois=[2000,5] [batch_inds,xmin,ymin,xmax,ymax] batch_inds全0 2000在cfg中调整RPN_POST_NMS_TOP_N
            rpn_scores=[2000] clsc层前景的softmax值
            rois是对rpn_rois区分前背景后筛选出batch_size个，并且前景在前，背景在后重新排列
            roi_scores是对rpn_scores的相同处理
            labels是batch_size个区域的标签，前景区域的标签与IOU最大的GT的标签相同，背景的标签为0
            bbox_targets=[batch_size，num_class*4] 正确类别的坐标为回归值，其余类别的坐标为0
            '''
            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            self._score_summaries.update(self._proposal_targets)

            return rois, roi_scores

    # 生成anchors
    def _anchor_component(self):
        # 生成每张图的anchor
        with tf.variable_scope('ANCHOR_' + 'default'):
            # just to get the shape right
            # 卷积特征图的尺寸 self._feat_stride[0]=16 经过4次pooling
            height = tf.to_int32(tf.ceil(self._im_info[0, 0] / np.float32(self._feat_stride[0])))
            width = tf.to_int32(tf.ceil(self._im_info[0, 1] / np.float32(self._feat_stride[0])))
            anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                                [height, width,
                                                 self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                                [tf.float32, tf.int32], name="generate_anchors")
            anchors.set_shape([None, 4])
            anchor_length.set_shape([])
            self._anchors = anchors
            self._anchor_length = anchor_length

    def build_network(self, sess, is_training=True):
        raise NotImplementedError

    # 对于回归的loss计算
    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        # 以rpn层为例：
        # bbox_pred  [b,h,w,A*4] reg层特征图
        # bbox_targets [1, height, width, A * 4] 整张图所有anchor的偏移量
        # sigma=3.0
        # dim=[1, 2, 3]
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        # tf.less(a,b) a<b返回真，否则返回假
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        # smoothL1_sign用于实现分段函数
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        # 求和并降为1维
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_box

    def _add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('loss_' + self._tag):
            # RPN, class loss
            # [1,b*A*h,w,2] cls层特征图的reshape
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
            # rpn_labels=[1, 1, A * height, width] 整张图所有预设anchor的标签 0为负样本 1为正样本 -1为无效样本
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            # 取出有效样本
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            # 获得有效样本的预测得分
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            # 获得有效样本的标签
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            # 计算交叉熵损失 RPN_BATCH_SIZE个anchor的均值
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

            # RPN, bbox loss
            # rpn_bbox_pred=[b,h,w,A*4] reg层特征图
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            # rpn_bbox_targets=[1, height, width, A * 4] 整张图所有anchor的偏移量
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            # 权重rpn_bbox_inside_weights=__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
            # inside_weights全是1，没起作用，outside_weights为有效样本数量的倒数，起到在一个batch内取平均的作用
            # rpn每次只处理一张图片，即一个batch，在一张图片上又取了RPN_BATCH_SIZE个anchor，作用就是对这些anchor取平均
            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

            # RCNN, class loss
            # RCNN部分，也就是RPN层之后的部分的分类损失
            cls_score = self._predictions["cls_score"]
            label = tf.reshape(self._proposal_targets["labels"], [-1])

            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=tf.reshape(cls_score, [-1, self._num_classes]), labels=label))

            # RCNN, bbox loss
            # 边框回归损失
            bbox_pred = self._predictions['bbox_pred']
            bbox_targets = self._proposal_targets['bbox_targets']
            # bbox_inside_weights = __C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
            # bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
            # inside_weights，outside_weights都是1，没起到作用
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']

            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

            self._losses['cross_entropy'] = cross_entropy
            self._losses['loss_box'] = loss_box
            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box

            loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
            # 总损失
            self._losses['total_loss'] = loss

            self._event_summaries.update(self._losses)

        return loss

    def create_architecture(self, sess, mode, num_classes, tag=None, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        self._image = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[self._batch_size, 3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self._tag = tag

        self._num_classes = num_classes  # 类别数
        self._mode = mode  # 模式，nms或top
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)

        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)

        self._num_anchors = self._num_scales * self._num_ratios

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        assert tag != None

        # handle most of the regularizer here
        # 正则化参数
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.FLAGS.weight_decay)
        if cfg.FLAGS.bias_decay:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        # 获得模型的输出
        with arg_scope([slim.conv2d, slim.conv2d_in_plane,
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0)):
            rois, cls_prob, bbox_pred = self.build_network(sess, training)

        layers_to_output = {'rois': rois}
        layers_to_output.update(self._predictions)

        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        if mode == 'TEST':
            # np.title 将矩阵横向复制self._num_classes次
            stds = np.tile(np.array(cfg.FLAGS2["bbox_normalize_stds"]), (self._num_classes))
            means = np.tile(np.array(cfg.FLAGS2["bbox_normalize_means"]), (self._num_classes))
            # 对框进行修正
            self._predictions["bbox_pred"] *= stds
            self._predictions["bbox_pred"] += means
        else:
            self._add_losses()
            layers_to_output.update(self._losses)

        val_summaries = []
        with tf.device("/cpu:0"):
            # 'GROUND_TRUTH'目录下添加标注后的图片
            val_summaries.append(self._add_image_summary(self._image, self._gt_boxes))
            # _event_summaries 包含 self._losses
            for key, var in self._event_summaries.items():
                # 添加标量
                val_summaries.append(tf.summary.scalar(key, var))
            # self._score_summaries 包含self._anchor_targets、self._proposal_targets、self._predictions
            for key, var in self._score_summaries.items():
                # 'SCORE/'目录下添加直方图
                self._add_score_summary(key, var)
            # self._act_summaries rpn_conv/3x3的输出特征图
            for var in self._act_summaries:
                # 'ACT/'目录下添加直方图
                self._add_act_summary(var)
            # self._train_summaries 包含所有可训练变量
            for var in self._train_summaries:
                # 'TRAIN/'目录下添加直方图
                self._add_train_summary(var)

        self._summary_op = tf.summary.merge_all()
        if not testing:
            self._summary_op_val = tf.summary.merge(val_summaries)

        # 模型输出，包含'rois'，self._losses，self._predictions
        return layers_to_output

    def get_variables_to_restore(self, variables, var_keep_dic):
        raise NotImplementedError

    def fix_variables(self, sess, pretrained_model):
        raise NotImplementedError

    # Extract the head feature maps, for example for vgg16 it is conv5_3
    # only useful during testing mode
    def extract_head(self, sess, image):
        feed_dict = {self._image: image}
        feat = sess.run(self._layers["head"], feed_dict=feed_dict)
        return feat

    # only useful during testing mode
    # 输入是仅包含单张图片的blob，用于测试网络
    def test_image(self, sess, image, im_info):
        feed_dict = {self._image: image,
                     self._im_info: im_info}
        cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                         self._predictions['cls_prob'],
                                                         self._predictions['bbox_pred'],
                                                         self._predictions['rois']],
                                                        feed_dict=feed_dict)
        return cls_score, cls_prob, bbox_pred, rois

    # 计算summary 训练过程中进行验证时使用
    def get_summary(self, sess, blobs):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary

    # 不包含summary的训练op 正常训练时使用
    def train_step(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                            self._losses['rpn_loss_box'],
                                                                            self._losses['cross_entropy'],
                                                                            self._losses['loss_box'],
                                                                            self._losses['total_loss'],
                                                                            train_op],
                                                                           feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

    # 计算summary的训练op 满足保存summary的间隔，需要保存summary时使用
    def train_step_with_summary(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                     self._losses['rpn_loss_box'],
                                                                                     self._losses['cross_entropy'],
                                                                                     self._losses['loss_box'],
                                                                                     self._losses['total_loss'],
                                                                                     self._summary_op,
                                                                                     train_op],
                                                                                    feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

    # 无返回值的训练op
    def train_step_no_return(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        sess.run([train_op], feed_dict=feed_dict)
