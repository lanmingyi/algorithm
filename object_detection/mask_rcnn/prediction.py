from object_detection.mask_rcnn.mrcnn.config import Config
from object_detection.mask_rcnn.mrcnn import model as modellib, utils
from object_detection.mask_rcnn.mrcnn import visualize
from object_detection.mask_rcnn.timer import Timer
import os
import sys
import cv2
import skimage.io
import time
import random

# Root directory of the project.
ROOT_DIR = os.path.abspath("../../")
print('ROOT_DIR', ROOT_DIR)
WEIGHTS_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.h5")
print('WEIGHTS_PATH', WEIGHTS_PATH)


# Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library


class MaskRCNNConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "MaskRCNN"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    # NUM_CLASSES = 1 + 2  # Background + classes
    NUM_CLASSES = 1 + 80  # Background + classes

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


def run():
    config = MaskRCNNConfig()
    # Create model
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir='models')
    print('model', model)
    # model.load_weights('models/mask_rcnn_wtott_0029.h5', by_name=True)
    model.load_weights(WEIGHTS_PATH, by_name=True)
    # class_names = ['BG', 'WT', 'OTT']
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    # IMAGE_DIR = 'datasets/test'
    IMAGE_DIR = 'images'
    # IMAGE_NAME = '12283150_12d37e6389_z.jpg'
    IMAGE_NAME = '3627527276_6fe8cd9bfe_z.jpg'
    # images = cv2.imread('datasets/test/2.jpg')
    images = skimage.io.imread(os.path.join(IMAGE_DIR, IMAGE_NAME))
    time1 = time.time()
    results = model.detect([images])
    # print(results[0])
    time2 = time.time()
    print('用时： {:.3f}s '.format(time2 - time1))

    r = results[0]
    print('分数：{}'.format(r['scores']))
    # print('类别：{}'.format(class_names[class_ids]))
    visualize.display_instances(images, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'], title='识别结果')

    # # Load a random image from the images folder
    # file_names = next(os.walk(IMAGE_DIR))[2]
    #
    # for x in range(len(file_names)):
    #     image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    #
    #     results = model.detect([image], verbose=1)
    #
    #     r = results[0]
    #     visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                                 class_names, r['scores'])
    return 0


if __name__ == "__main__":
    run()
