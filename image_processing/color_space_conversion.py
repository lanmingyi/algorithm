import cv2
import numpy as np


# ################### 转换颜色空间 #####################
# 使用函数cv2.cvtColor(input_image ，flag)
# flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
# print(flags)  # 可以使用的flags
# print(dir(cv2))
# HSV空间中，H表示色彩/色度，取值范围 [0，179]，S表示饱和度，取值范围 [0，255]，V表示亮度，取值范围 [0，255]。


# ####################  跟踪物体 #########################
# HSV空间比在BGR空间中更容易表示一个特定的颜色。现在要提取蓝色物体
# 步骤如下:从视频提取每一帧,将图像转换为HSV空间,设置HSV阈值到蓝色范围,获取蓝色物体，可以对物体操作，比如画圈
# cap = cv2.VideoCapture(0)
#
# while 1:
#     ret, frame = cap.read()  # 读取视频帧
#     # print(ret, frame)
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # HSV空间
#
#     lower_blue = np.array([110, 50, 50])  # 设定蓝色的阈值,阈值下限
#     upper_blue = np.array([130, 255, 255])  # 阈值上限
#
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)  # 设定取值范围
#     res = cv2.bitwise_and(frame, frame, mask=mask)  # 对原图像处理
#
#     cv2.imshow('frame', frame)
#     cv2.imshow('mask', mask)
#     cv2.imshow('res', res)
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
# cv2.destroyAllWindows()


# ############ 如何找到要跟踪对象的HSV值 #####################
# 可以使用这个函数 cv2.cvtColor()，但是需要输入的参数是BGR值，而不是一副图片。
# green = np.uint8([[[0, 255, 0]]])  #
# print('秩：', green.ndim)  # 秩，即轴的个数。维数
# print('维度：', green.shape)  # 维度，即行数、列数、通道数
# print('元素总个数：', green.size)  # 元素总个数
# print('元素类型：', green.dtype)  # 元素类型
# print('元素字节大小：', green.itemsize)  # 元素字节大小

# hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
# print(hsv_green)  # 要找到绿色的HSV值


# ########### 同时提取红绿蓝三个不同颜色的物体 ##################
img = cv2.imread("3.jpg")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # HSV空间

lower_blue = np.array([110, 100, 100])  # blue
upper_blue = np.array([130, 255, 255])

lower_green = np.array([60, 100, 100])  # green
upper_green = np.array([70, 255, 255])

lower_red = np.array([0, 100, 100])  # red
upper_red = np.array([10, 255, 255])

# 利用opencv的inRange()函数，制作掩模，再用bitwise_and()函数，提取感兴趣区域：
red_mask = cv2.inRange(hsv, lower_red, upper_red)  # 取红色
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)  # 蓝色
green_mask = cv2.inRange(hsv, lower_green, upper_green)  # 绿色
cv2.imshow('red_mask', red_mask)

# 对原图像处理
# bitwise_and是对二进制数据进行“与”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“与”操作，1&1=1，1&0=0，0&1=0，0&0=0
# bitwise_or是对二进制数据进行“或”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“或”操作，1|1=1，1|0=0，0|1=0，0|0=0
# bitwise_xor是对二进制数据进行“异或”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“异或”操作，1^1=0,1^0=1,0^1=1,0^0=0
# bitwise_not是对二进制数据进行“非”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“非”操作，~1=0，~0=1
red = cv2.bitwise_and(img, img, mask=red_mask)
green = cv2.bitwise_and(img, img, mask=green_mask)
blue = cv2.bitwise_and(img, img, mask=blue_mask)
cv2.imshow('blue', blue)

res = green+red+blue
cv2.imshow('img', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
