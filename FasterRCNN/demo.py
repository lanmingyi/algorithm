#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
# 在一个窗口显示所有类别标注:代码改动也不复杂，就是把vis_detections函数中for循环前后三行代码移动到demo函数的for循环前后。
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
# 计算mAP的值
# from lib.utils.test import test_net
from lib.datasets.factory import get_imdb

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
# CLASSES = ('__background__','dog')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_40000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}



# def vis_detections(im, class_name, dets, thresh=0.5):
def vis_detections(im, class_name, dets, ax, thresh=0.5):  # 增加ax参数
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]  # 返回置信度大于阈值的窗口下标
    if len(inds) == 0:
        return

    # im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]  # 人脸坐标位置（Xmin,Ymin,Xmax,Ymax）
        score = dets[i, -1]  # 置信度得分

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)  # 矩形线宽从3.5改为1
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.draw()


def demo(sess, net, image_name):
    # 检测目标类，在图片中提议窗口
    """Detect object classes in an image using pre-computed object proposals."""
    # print(sess)
    # print(net)
    # Load the demo image，得到图片绝对地址
    im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)  # 拼接路径，返回'A/B/C'之类路径
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()  # time.time()返回当前时间
    timer.tic()  # 返回开始时间，见'time.py'中
    scores, boxes = im_detect(sess, net, im)  # 检测，返回得分和人脸区域所在位置
    timer.toc()  # 返回平均时间，'time.py'文件中
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8   # 0.1、0.8
    NMS_THRESH = 0.3    # 0.1、0.3
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for cls_ind, cls in enumerate(CLASSES[1:]):  # enumerate:用于遍历序列中元素及他们的下标
        cls_ind += 1  # because we skipped background，cls_ind:下标，cls:元素
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]  # 返回当前坐标
        cls_scores = scores[:, cls_ind]  # 返回当前得分
        # hstack:拷贝，合并参数
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        # vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)
        # 画检测框
        vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)  # 将ax做为参数传入vis_detections
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    # output ='D:\Python\ArtificialIntelligence\ImageRecognition\FasterRCNNTFWindows\output'
    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], NETS[demonet][0])
    # tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    # tfconfig.gpu_options.allow_growth = True
    tfconfig.gpu_options.allow_growth = False


    # init session
    sess = tf.Session(config=tfconfig)

    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    # elif demonet == 'res101':
        # net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError

    n_classes = len(CLASSES)
    # create the structure of the net having a certain shape (which depends on the number of classes) 
    net.create_architecture(sess, "TEST", n_classes,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)  # 重载模型的参数，继续训练或用于测试数据。

    print('Loaded network {:s}'.format(tfmodel))

    # im_names = ['000456.jpg', '000457.jpg', '000542.jpg', '001150.jpg',
    #             '001763.jpg', '004545.jpg']
    im_names = ['004545.jpg']
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)  # 逐个跑demo
    # # 计算mAP值
    # imdb = get_imdb("voc_2007_trainval")
    # test_net(sess, net, imdb, 'default')
    plt.show()
