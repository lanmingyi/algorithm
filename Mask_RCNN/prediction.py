from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from timer import Timer
import os
import cv2
import skimage.io

import time
import random

class WTOTTConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "wtott"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + classes

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


config = WTOTTConfig()
# Create model
model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir='models')
# model.load_weights('models/mask_rcnn_wtott_0029.h5', by_name=True)
model.load_weights('models/mask_rcnn_coco.h5', by_name=True)

class_names = ['BG', 'WT', 'OTT']

# IMAGE_DIR = 'datasets/test'
IMAGE_DIR = 'images'
IMAGE_NAME = '12283150_12d37e6389_z.jpg'
# images = cv2.imread('datasets/test/2.jpg')
images = skimage.io.imread(os.path.join(IMAGE_DIR, IMAGE_NAME))
time1 = time.time()
results = model.detect([images])
# print(results[0])
time2 = time.time()
print('用时： {:.3f}s '.format(time2-time1))

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
