# coding=utf-8
# author:兰明易 time:2020-05-03
import cv2
import numpy as np
import pandas as pd
# import PIL.Image as img
from PIL import Image

# 漫画风格
# # img_rgb = cv2.imread('5.jpg')
# img_rgb = cv2.imdecode(np.fromfile("D:/创业/图像处理/1.jpg", dtype=np.uint8), -1)  # 读取中文路径
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
# # print(img_gray)
# img_gray = cv2.medianBlur(img_gray, 5)  # 对图像模糊化
# # 使用adaptiveThreshold()方法对图片进行二值化操作
# img_edge = cv2.adaptiveThreshold(img_gray, 255,
#                                  cv2.ADAPTIVE_THRESH_MEAN_C,
#                                  cv2.THRESH_BINARY, blockSize=3, C=2)
# # cv2.imshow('image',img_edge)
# # cv2.waitKey()
# # cv2.imwrite('11-1.jpg', img_edge)
# cv2.imencode('.jpg', img_edge)[1].tofile("D:/创业/图像处理/"+"1-2.jpg")  # 中文路径

# # 写实风格
# # img_rgb = cv2.imread('5.jpg')
# img_rgb = cv2.imdecode(np.fromfile("D:/创业/图像处理/1.jpg", dtype=np.uint8), -1)  # 读取中文路径
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
# # 使用高斯滤波进行图片模糊化。ksize表示高斯核的大小，sigmaX和sigmaY分别表示高斯核在 X 和 Y 方向上的标准差
# img_blur = cv2.GaussianBlur(img_gray, ksize=(21, 21),
#                             sigmaX=0, sigmaY=0)
# img_edge = cv2.divide(img_gray, img_blur, scale=245)  # 使用cv2.divide()方法对原图和模糊图像进行融合
# # cv2.imwrite('5-3.jpg', img_edge)
# cv2.imencode('.jpg', img_edge)[1].tofile("D:/创业/图像处理/"+"1-7.jpg")  # 中文路径


# # 将两张图片合并为一张
# img1 = cv2.imread('2.jpg')
# img2 = cv2.imread('2-2.jpg')
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# # ====使用numpy的数组矩阵合并concatenate======
#
# image = np.concatenate((gray1, gray2))
# # 纵向连接 image = np.vstack((gray1, gray2))
# # 横向连接 image = np.concatenate([gray1, gray2], axis=1)
# # image = np.array(df)  # dataframe to ndarray
# cv2.imshow('image', image)
# cv2.imwrite('2-4.jpg', image)
# cv2.waitKey()


# 截取图片中的一部分并且复制到另一张图片中
# IMG = '3.jpg'  # 图片地址
# img = Image.open(IMG)  # 用PIL打开一个图片
# print(img.size)
# box = (0, 0, 200, 200)  # box代表需要剪切图片的位置格式为:xmin ymin xmax ymax
# ng = img.crop(box)  # 对im进行裁剪 保存为ng(这里im保持不变)
# ng = ng.rotate(20)  # ng为裁剪出来的图片，进行向左旋转20度 向右为负数
# # ng.show()
# # ng.save('C:\\Users\\Ilearn\\Desktop\\temp\\copy.JPG')
# img.paste(ng, (0 + 50, 0 + 50))  # 将ng复制到im上，放入的位置为(3664 + 50, 2193 + 50)
# # img.save('C:\\Users\\Ilearn\\Desktop\\temp\\transform.JPG')  # 保存变化后的图片
# img.show()
# # 图片旋转
# im = Image.open("3.jpg")
# im_45 = im.rotate(45)
# im_30 = im.rotate(30, Image.NEAREST, 1)
# print(im_45.size,im_30.size)
# im_45.show()
# # im_30.show()
