"""
Copyright (c) 2021 Bright Mind. All rights reserved.
Written by MingYi Lan
"""

import cv2
import numpy as np
from PIL import ImageFont, Image, ImageDraw


image = cv2.imread('data/COCO_val2014_000000000338.jpg')
font = ImageFont.truetype("data/simhei.ttf", 42)
image = Image.fromarray(image)
draw = ImageDraw.Draw(image)

label = "开始时间: "
left, top = 30, 30
draw.text((int(left), int(top)), label, tuple([0, 0, 255]), font=font)
image = np.asarray(image)
cv2.imshow('字体', image)

