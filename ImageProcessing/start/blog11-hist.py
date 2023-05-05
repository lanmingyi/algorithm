#encoding:utf-8
import cv2  
import numpy as np
import matplotlib.pyplot as plt

# src = cv2.imread('Lena.png')
# histb = cv2.calcHist([src], [0], None, [256], [0,255])
# histg = cv2.calcHist([src], [1], None, [256], [0,255])
# histr = cv2.calcHist([src], [2], None, [256], [0,255])
# cv2.imshow("src", src)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.plot(histb, color='b')
# plt.plot(histg, color='g')
# plt.plot(histr, color='r')
# plt.show()


# src = cv2.imread('Lena.png')
# #参数:原图像 通道[0]-B 掩码 BINS为256 像素范围0-255
# hist = cv2.calcHist([src], [0], None, [256], [0,255])
# print(type(hist))
# print(hist.size)
# print(hist.shape)
# print(hist)


# #绘制sin函数曲线
# x1 = np.arange(0, 6, 0.1)
# y1 = np.sin(x1)
# plt.plot(x1, y1)
# #绘制坐标点折现
# x2 = [0, 1, 2, 3, 4, 5, 6]
# y2 = [0.3, 0.4, 2.5, 3.4, 4, 5.8, 7.2]
# plt.plot(x2, y2)
# #省略有规则递增的x2参数
# y3 = [0, 0.5, 1.5, 2.4, 4.6, 8]
# plt.plot(y3, color="r")
# plt.show()


# src = cv2.imread('Lena.png')
# histb = cv2.calcHist([src], [0], None, [256], [0,255])
# histg = cv2.calcHist([src], [1], None, [256], [0,255])
# histr = cv2.calcHist([src], [2], None, [256], [0,255])
# cv2.imshow("src", src)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.plot(histb, color='b')
# plt.plot(histg, color='g')
# plt.plot(histr, color='r')
# plt.show()
