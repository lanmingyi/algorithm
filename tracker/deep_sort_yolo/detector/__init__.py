from .YOLOv3 import YOLOv3


__all__ = ['build_detector']  # __all__ 限制from * import *中import的包名。
"""
'YOLOV3': {'CFG': './detector/YOLOv3/cfg/yolo_v3.cfg', 
        'WEIGHT': './detector/YOLOv3/weight/yolov3.weights', 'CLASS_NAMES': './detector/YOLOv3/cfg/coco.names', 
        'SCORE_THRESH': 0.5, 'NMS_THRESH': 0.4, 
"""


def build_detector(cfg, use_cuda):
    return YOLOv3(cfg.YOLOV3.CFG, cfg.YOLOV3.WEIGHT, cfg.YOLOV3.CLASS_NAMES, 
                    score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH, 
                    is_xywh=True, use_cuda=use_cuda)
