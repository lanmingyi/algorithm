"""
Mask R-CNN
Base Configurations class.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import numpy as np


# Base Configuration Class。基础配置类
# Don't use this class directly. Instead, sub-class it and override
# 不要直接使用这个类。相反，子类化它并覆盖（重载）它
# the configurations you need to change. 你需要更改的配置

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # 名称的配置。例如：COCO,实验3等等
    # Useful if your code needs to do things differently depending on which experiment is running.
    # 如果你的代码需要根据运行的实验进行不同的操作，那么这个功能非常有用。
    NAME = None  # Override in sub-classes

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically handle 2 images of 1024x1024px.
    # 在每个GPU上训练的图像数量，一个12GB的GPU通常可以处理2张1024×1024px的图像
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    # 调整基于你的GPU内存和图像大小。使用你的GPU可以处理的最大的数字以获得最佳的性能
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch。每个epoch的训练步骤数
    # This doesn't need to match the size of the training set. 这并不需要匹配训练集的大小
    # Tensorboard updates are saved at the end of each epoch, 在每个历元结束的时候会保存更新
    # so setting this to a smaller number means getting more frequent TensorBoard updates.
    # 所以设置这个为一个较小的数字意味着得到更频繁的张力板更新
    # Validation stats are also calculated at each epoch end and they might take a while,
    # 验证状态也在每个epoch结束时计算，他们可能需要一段时间
    # so don't set this too small to avoid spending a lot of time on validation stats.
    # 所以不要将其设置的太小，以免在验证状态上花费大量时间
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # 在每个训练阶段结束时运行的验证步骤数
    # A bigger number improves accuracy of validation stats, but slows down the training.
    # 更大的数字可以提高验证状态的准确性，但会降低训练速度
    VALIDATION_STEPS = 50

    # Backbone network architecture。主干网体系结构
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature of model.resnet_graph.
    # If you do so, you need to supply a callable to COMPUTE_BACKBONE_SHAPE as well。
    # callable：可调用的，signature：签名
    BACKBONE = "resnet101"

    # Only useful if you supply a callable to BACKBONE. 只有当你提供一个可调用到主干时才有用
    # Should compute the shape of each layer of the FPN Pyramid. 应该计算FPN金字塔各层的形状
    # See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = None

    # The strides of each layer of the FPN Pyramid. These values are based on a Resnet101 backbone.
    # FPN金字塔各层的步幅，这些值是基于Resnet101主干的
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Size of the fully-connected layers in the classification graph
    # 分类图中全连通层的尺寸
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Size of the top-down layers used to build the feature pyramid
    # 用于构建特征金字塔的自顶向下的层的大小
    TOP_DOWN_PYRAMID_SIZE = 256

    # Number of classification classes (including background)。分类类别数目
    NUM_CLASSES = 1  # Override in sub-classes

    # Length of square anchor side in pixels。正方形锚边的长度，以像素为单位
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)。每个单元锚点的比率
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    # 值1表示正方形锚点， 0.5表示宽锚点
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride 步幅
    # If 1 then anchors are created for each cell in the backbone feature map.
    # 为主干特征图中的每个单元创建锚
    # If 2, then anchors are created for every other cell, and so on.
    # 为每一个其他单元创建锚，以此类推
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # 非最大抑制阈值筛选RPN建议框
    # You can increase this during training to generate more proposals.
    # 你可以在训练中增加这一点，以产生更多的proposals
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    # 对于RPN训练，每个图像需要使用多少个锚点
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    
    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    # ROIs在tf.nn.top_k之后和非最大抑制之前
    PRE_NMS_LIMIT = 6000

    # ROIs kept after non-maximum suppression (training and inference)
    # 在非最大抑制（训练和推断）后保持ROIs
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce memory load.
    # 如果启用，则将实例掩膜调整为较小的大小，以减少内存负载
    # Recommended when using high-resolution images. 当使用高分辨率图像时推荐
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing。输入图像调整
    # Generally, use the "square" resizing mode for training and predicting and it should work well in most cases.
    # 一般来说，使用方形大小调整模式进行训练和预测，它应该在大多情况下工作良好。
    # In this mode, images are scaled up such that the small side is = IMAGE_MIN_DIM,
    # 在这种模式下，图像按比例放大，使小的边为IMAGE_MIN_DIM
    # but ensuring that the scaling doesn't make the long side > IMAGE_MAX_DIM.
    # 但要确保缩放不会使长的边大于IMAGE_MAX_DIM
    # Then the image is padded with zeros to make it a square so multiple images can be put in one batch.
    # 然后用零填充图像，使它成为一个正方形，这样多个图像可以放在一个批处理。
    # Available resizing modes: 可以调整模式
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales up before padding.
    #         如果图像最小亮度或图像最小比例不是没有，那么它在填充前会放大
    #         IMAGE_MAX_DIM is ignored in this mode. 在此模式下忽略图像最大亮度
    #         The multiple of 64 is needed to ensure smooth scaling of feature maps up and down the 6 levels of
    #         the FPN pyramid (2**6=64)。64的倍数需要保证FPN金字塔的6层中feature map的平化上下缩放
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         从图像中随机选取裁剪(作物)。首先根据图像最小DIM(亮度)和图像最小尺度对图像进行缩放，然后随机抽取大小为
    #         IMAGE_MIN_DIM x IMAGE_MIN_DIM的图像进行裁剪。仅用于训练
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # Minimum scaling ratio. 最小缩放比例
    # Checked after MIN_IMAGE_DIM and can force further up scaling.
    # For example, if set to 2 then images are scaled up to double the width and height, or more,
    # even if MIN_IMAGE_DIM doesn't require it.
    # However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0
    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # 每幅图像的彩色通道数
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3

    # Image mean (RGB)。图像均值
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # 每幅图像需要提供给分类器/mask头的ROIs数
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative ratio of 1:3.
    # Mask RCNN论文使用512，但通常RPN没有产生足够的积极建议框来填补这一空白，并保持1：3的积极:消极比率
    # You can increase the number of proposals by adjusting the RPN NMS threshold.
    # 你可以通过调整RPN NMS阈值来增加proposals的数量
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads。用于训练分类器/mask头的积极ROIs百分比
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs。池化的ROIs，合并的ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask。输出掩膜形状
    # To change this you also need to change the neural network mask branch
    # 要改变这一点，你还需要改变神经网络的掩膜分支
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    # 在一个图像中使用的最大ground truth实例数
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    # RPN和最终检测的边界框细化标准偏差
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections。最终检测的最大数量
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance。接受检测到实例的最小概率值
    # ROIs below this threshold are skipped。低于此阈值的ROIs将被跳过
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection。用于检测的非最大抑制阈值
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum。学习率和动力
    # The Mask RCNN paper uses lr=0.02,
    # but on TensorFlow it causes weights to explode. 但在TensorFlow中它会导致权重爆炸
    # Likely due to differences in optimizer。可能是由于优化器的不同
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization。权重衰变正则
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.更精确的优化损失权重
    # Can be used for R-CNN training setup.可用于R-CNN训练设置
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    # Use RPN ROIs or externally generated ROIs for training
    # 使用RPN ROIs或外部生成的ROIs进行训练
    # Keep this True for most situations. 在大多数情况下都要保持True
    # Set to False if you want to train the head branches on ROI generated by code rather than the ROIs from the RPN.
    # 如果你希望根据代码生成的ROIs(而不是RPN生成的ROIs)训练head分支，则设置为False
    # For example, to debug the classifier head without having to train the RPN.
    # 例如，调试分类器头而不需要训练RPN
    USE_RPN_ROIS = True

    # Train or freeze batch normalization layers。训练或冻结批处理标准化层
    #     None: Train BN layers. This is the normal mode。训练BN层，这是正常模式
    #     False: Freeze BN layers. Good when using a small batch size。冻结BN层，当使用小批量大小
    #     True: (don't use). Set layer in training mode even when predicting。在训练模式设置层，即使在预测时
    TRAIN_BN = False  # Defaulting to False since batch size is often small。默认为False，因为批大小通常较小

    # Gradient norm clipping。梯度标准裁剪
    GRADIENT_CLIP_NORM = 5.0

    def __init__(self):
        """Set values of computed attributes.设置计算属性的值"""
        # Effective batch size。有效的批量大小
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                self.IMAGE_CHANNEL_COUNT])

        # Image meta data length。图像元数据长度
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def display(self):
        """Display Configuration values.显示配置值"""
        print("\nConfigurations:")
        # dir()是一个内置函数，用于列出对象的所有属性及方法。
        for a in dir(self):
            # callable() 函数用于检查一个对象是否是可调用的。函数是可调用的
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


# con = Config()
# con.display()
