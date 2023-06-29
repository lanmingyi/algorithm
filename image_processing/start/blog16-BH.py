# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt


# # 读取原始图像
# img = cv2.imread('miao.jpg')
# # 图像灰度转换
# grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # 获取图像高度和宽度
# height = grayImage.shape[0]
# width = grayImage.shape[1]
# # 创建一幅图像
# result = np.zeros((height, width), np.uint8)
# # 图像灰度非线性变换：DB=DA×DA/255
# for i in range(height):
#     for j in range(width):
#         gray = int(grayImage[i, j]) * int(grayImage[i, j]) / 255
#         result[i, j] = np.uint8(gray)
# # 显示图像
# cv2.imshow("Gray Image", grayImage)
# cv2.imshow("Result", result)
# # 等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 绘制曲线
# def log_plot(c):
#     x = np.arange(0, 256, 0.01)
#     y = c * np.log(1 + x)
#     plt.plot(x, y, 'r', linewidth=1)
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
#     plt.title(u'对数变换函数')
#     plt.xlim(0, 255), plt.ylim(0, 255)
#     plt.show()
#
#
# # 对数变换
# def log(c, img):
#     output = c * np.log(1.0 + img)
#     output = np.uint8(output + 0.5)
#     return output
#
#
# # 读取原始图像
# img = cv2.imread('2016.png')
# # 绘制对数变换曲线
# log_plot(42)
# # 图像灰度对数变换
# output = log(42, img)
# # 显示图像
# cv2.imshow('Input', img)
# cv2.imshow('Output', output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# def gamma_plot(c, v):
#     x = np.arange(0, 256, 0.01)
#     y = c * x ** v
#     plt.plot(x, y, 'r', linewidth=1)
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
#     plt.title(u'伽马变换函数')
#     plt.xlim([0, 255]), plt.ylim([0, 255])
#     plt.show()
#
#
# # 伽玛变换
# def gamma(img, c, v):
#     lut = np.zeros(256, dtype=np.float32)
#     for i in range(256):
#         lut[i] = c * i ** v
#     output_img = cv2.LUT(img, lut)  # 像素灰度值的映射
#     output_img = np.uint8(output_img + 0.5)
#     return output_img
#
#
# # 读取原始图像
# img = cv2.imread('2019.png')
# # 绘制伽玛变换曲线
# gamma_plot(0.00000005, 4.0)
# # 图像灰度伽玛变换
# output = gamma(img, 0.00000005, 4.0)
# # 显示图像
# cv2.imshow('Imput', img)
# cv2.imshow('Output', output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
