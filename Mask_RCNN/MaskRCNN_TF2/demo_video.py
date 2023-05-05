

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
import sys
import cv2
import colorsys
import random
import numpy as np
from timeit import default_timer as timer
from PIL import Image, ImageFont, ImageDraw

from mrcnn import utils
import mrcnn.model as modellib

ROOT_DIR = os.path.abspath("./")
# Import COCO config To find local version
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import coco
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU为增长式占用
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 打印异常
        print(e)
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# # 在session内增加config = config
# sess = tf.compat.v1.Session(config=config)

# config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
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


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def predict():
    config = InferenceConfig()
    config.display()
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    return model


def detect_video(model, video_path, output_path=""):
    vid = cv2.VideoCapture(video_path)
    # vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    while True:
        return_value, frame = vid.read()
        if return_value == False:
            break
        img = Image.fromarray(frame)
        image1 = np.asarray(img)
        # detect object
        image2 = apply_rect(image1, model)
        # play
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", image2)
        #if isOutput:
        #    out.write(image2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.waitKey(0)


def apply_rect(image, model):
    # return image
    # Run detection
    results = model.detect([image], verbose=1)
    r = results[0]
    # print(r)
    boxes = r['rois']
    #masks = r['masks']
    class_ids = r['class_ids']

    N = boxes.shape[0]
    N = boxes.shape[0]
    for i in range(N):
        color = (0, 255, 0)
        y1, x1, y2, x2 = boxes[i]
        left = x1  # left
        top = y1   # top
        right = x2  # right
        bottom = y2  # bottom
        # Box (left, top), (right, bottom)
        cv2.rectangle(image, (left, top), (right, bottom), color=color, thickness=1)
        # Label
        class_id = class_ids[i]
        label = class_names[class_id]
        cv2.putText(image, label, (left, top-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return image


if __name__ == '__main__':
    model = predict()
    videofile = 'D:/LMY_Code/MaskRCNN/datasets/videos/demo2.mp4'
    # videofile = 'C:/tmp/cars.mp4'
    detect_video(model, video_path=videofile)
