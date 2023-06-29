"""
Copyright (c) 2021 Bright Mind. All rights reserved.
Written by MingYi Lan
"""
import sys
sys.path.append("detectors/yolo")

# detector.py文件
from .detector import YOLOv3
__all__ = ['YOLOv3']