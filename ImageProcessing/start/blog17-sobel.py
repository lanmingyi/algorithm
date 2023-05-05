# -*- coding: utf-8 -*-
import cv2  
import numpy as np  
import matplotlib.pyplot as plt

# # 读取图像
# img = cv2.imread('lena.png')
# lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # 灰度化处理图像
# grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # Roberts算子
# kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
# kernely = np.array([[0, -1], [1, 0]], dtype=int)
# x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
# y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
# # 转uint8
# absX = cv2.convertScaleAbs(x)
# absY = cv2.convertScaleAbs(y)
# Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# # 用来正常显示中文标签
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 显示图形
# titles = [u'原始图像', u'Roberts算子']
# images = [lenna_img, Roberts]
# for i in range(2):
#    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
#    plt.title(titles[i])
#    plt.xticks([]), plt.yticks([])
# plt.show()


# # 读取图像
# img = cv2.imread('lena.png')
# lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # 灰度化处理图像
# grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # Prewitt算子
# kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
# kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
# x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
# y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
# # 转uint8
# absX = cv2.convertScaleAbs(x)
# absY = cv2.convertScaleAbs(y)
# Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# # 用来正常显示中文标签
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 显示图形
# titles = [u'原始图像', u'Prewitt算子']
# images = [lenna_img, Prewitt]
# for i in range(2):
#    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
#    plt.title(titles[i])
#    plt.xticks([]), plt.yticks([])
# plt.show()


# # 读取图像
# img = cv2.imread('lena.png')
# lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # 灰度化处理图像
# grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # Sobel算子
# x = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)  # 对x求一阶导
# y = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1)  # 对y求一阶导
# absX = cv2.convertScaleAbs(x)
# absY = cv2.convertScaleAbs(y)
# Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# # 用来正常显示中文标签
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 显示图形
# titles = [u'原始图像', u'Sobel算子']
# images = [lenna_img, Sobel]
# for i in range(2):
#     plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()


# # 读取图像
# img = cv2.imread('lena.png')
# lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # 灰度化处理图像
# grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # 拉普拉斯算法
# dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=3)
# Laplacian = cv2.convertScaleAbs(dst)
# # 用来正常显示中文标签
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 显示图形
# titles = [u'原始图像', u'Laplacian算子']
# images = [lenna_img, Laplacian]
# for i in range(2):
#     plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()


# # 读取图像
# img = cv2.imread('lena.png')
# lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # 灰度化处理图像
# grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # 高斯滤波
# gaussianBlur = cv2.GaussianBlur(grayImage, (3, 3), 0)
# # 阈值处理
# ret, binary = cv2.threshold(gaussianBlur, 127, 255, cv2.THRESH_BINARY)
# # Roberts算子
# kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
# kernely = np.array([[0, -1], [1, 0]], dtype=int)
# x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
# y = cv2.filter2D(binary, cv2.CV_16S, kernely)
# absX = cv2.convertScaleAbs(x)
# absY = cv2.convertScaleAbs(y)
# Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# # Prewitt算子
# kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
# kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
# x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
# y = cv2.filter2D(binary, cv2.CV_16S, kernely)
# absX = cv2.convertScaleAbs(x)
# absY = cv2.convertScaleAbs(y)
# Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# # Sobel算子
# x = cv2.Sobel(binary, cv2.CV_16S, 1, 0)
# y = cv2.Sobel(binary, cv2.CV_16S, 0, 1)
# absX = cv2.convertScaleAbs(x)
# absY = cv2.convertScaleAbs(y)
# Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# # 拉普拉斯算法
# dst = cv2.Laplacian(binary, cv2.CV_16S, ksize=3)
# Laplacian = cv2.convertScaleAbs(dst)
# # 效果图
# titles = ['Source Image', 'Binary Image', 'Roberts Image',
#           'Prewitt Image', 'Sobel Image', 'Laplacian Image']
# images = [lenna_img, binary, Roberts, Prewitt, Sobel, Laplacian]
# for i in np.arange(6):
#     plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()