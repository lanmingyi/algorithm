"""
Mask R-CNN
Train on the 风力发电机(wind turbines(WT))、油罐车(oil tank truck（OTT）)

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see mrcnn/LICENSE for details)
Written by MINGYI LAN
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# 根目录
ROOT_DIR = os.path.abspath("")

# 导入 Mask RCNN 算法
# sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# 预训练权重文件的路径
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.h5")

# 模型训练保存路径
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "models")

############################################################
#  Configurations
############################################################


class WTOTTConfig(Config):
    """
        训练数据集的配置
    """
    # Give the configuration a recognizable name
    NAME = "wtott"

    # 一个GPU训练两张图片
    IMAGES_PER_GPU = 2

    # 识别类别数量
    NUM_CLASSES = 1 + 1  # Background + classes

    # 每个epoch的训练步数
    STEPS_PER_EPOCH = 100

    # 只检测可信度大于等于90%的
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class WTOTTDataset(utils.Dataset):

    def load_wtoot(self, dataset_dir, subset):
        """加载数据集
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        self.add_class("wtott", 1, "WT")
        # self.add_class("wtott", 2, "OTT")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # 加载注释
        # 格式:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        annotations = [a for a in annotations if a['regions']]

        # 加载图片
        for a in annotations:
            # 获得shape_attributes里x，y坐标
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() 需要将多边形转换为mask的图像大小。
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "wtott",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """为图像生成实例mask。
       Returns:
        masks: 一个包含形状[高度，宽度，实例计数]的bool数组。每个实例一个掩码。
        class_ids: 类别id的一维数组。
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "wtott":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        # classID = np.zeros((len(info["polygons"]),))  # 多类别
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
            # if p['name'] == 'WT':
            #     classID[i, ] = 1
            # else:
            #     classID[i, ] = 2

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

        # return mask.astype(np.bool), classID
        # print(np.array(classID, dtype=np.int32))
        # return mask.astype(np.bool), np.array(classID, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "wtott":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = WTOTTDataset()
    dataset_train.load_wtoot(os.path.join(ROOT_DIR, "images"), "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = WTOTTDataset()
    dataset_val.load_wtoot(os.path.join(ROOT_DIR, "images"), "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


############################################################
#  Training
############################################################

if __name__ == '__main__':
    # Configurations
    config = WTOTTConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

    weights = 'coco'
    # Select weights file to load
    if weights == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif weights == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif weights == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = weights

    # Load weights
    print("Loading weights ", weights_path)
    if weights == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    train(model)

