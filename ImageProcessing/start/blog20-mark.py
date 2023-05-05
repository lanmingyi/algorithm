# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

# # 读取原始图像
# img = cv2.imread('lena.png')
# # 获取图像高度和宽度
# height = img.shape[0]
# width = img.shape[1]
# # 创建一幅图像
# new_img = np.zeros((height, width, 3), np.uint8)
# # 图像量化操作 量化等级为2
# for i in range(height):
#     for j in range(width):
#         for k in range(3):  # 对应BGR三分量
#             if img[i, j][k] < 128:
#                 gray = 0
#             else:
#                 gray = 128
#             new_img[i, j][k] = np.uint8(gray)
#         # 显示图像
# cv2.imshow("src", img)
# cv2.imshow("", new_img)
# # 等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 读取原始图像
# img = cv2.imread('lena.png')
# # 获取图像高度和宽度
# height = img.shape[0]
# width = img.shape[1]
# # 创建一幅图像
# new_img1 = np.zeros((height, width, 3), np.uint8)
# new_img2 = np.zeros((height, width, 3), np.uint8)
# new_img3 = np.zeros((height, width, 3), np.uint8)
# # 图像量化等级为2的量化处理
# for i in range(height):
#     for j in range(width):
#         for k in range(3):  # 对应BGR三分量
#             if img[i, j][k] < 128:
#                 gray = 0
#             else:
#                 gray = 128
#             new_img1[i, j][k] = np.uint8(gray)
#
# # 图像量化等级为4的量化处理
# for i in range(height):
#     for j in range(width):
#         for k in range(3):  # 对应BGR三分量
#             if img[i, j][k] < 64:
#                 gray = 0
#             elif img[i, j][k] < 128:
#                 gray = 64
#             elif img[i, j][k] < 192:
#                 gray = 128
#             else:
#                 gray = 192
#             new_img2[i, j][k] = np.uint8(gray)
#
# # 图像量化等级为8的量化处理
# for i in range(height):
#     for j in range(width):
#         for k in range(3):  # 对应BGR三分量
#             if img[i, j][k] < 32:
#                 gray = 0
#             elif img[i, j][k] < 64:
#                 gray = 32
#             elif img[i, j][k] < 96:
#                 gray = 64
#             elif img[i, j][k] < 128:
#                 gray = 96
#             elif img[i, j][k] < 160:
#                 gray = 128
#             elif img[i, j][k] < 192:
#                 gray = 160
#             elif img[i, j][k] < 224:
#                 gray = 192
#             else:
#                 gray = 224
#             new_img3[i, j][k] = np.uint8(gray)
#
# # 用来正常显示中文标签
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 显示图像
# titles = [u'(a) 原始图像', u'(b) 量化-L2', u'(c) 量化-L4', u'(d) 量化-L8']
# images = [img, new_img1, new_img2, new_img3]
# for i in range(4):
#     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray'),
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()


# # 读取原始图像
# img = cv2.imread('flower.png')
# # 图像二维像素转换为一维
# data = img.reshape((-1, 3))
# data = np.float32(data)
# # 定义中心 (type,max_iter,epsilon)
# criteria = (cv2.TERM_CRITERIA_EPS +
#             cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# # 设置标签
# flags = cv2.KMEANS_RANDOM_CENTERS
# # K-Means聚类 聚集成4类
# compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
# # 图像转换回uint8二维类型
# centers = np.uint8(centers)
# res = centers[labels.flatten()]
# dst = res.reshape((img.shape))
# # 图像转换为RGB显示
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# # 用来正常显示中文标签
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 显示图像
# titles = [u'原始图像', u'聚类量化 K=4']
# images = [img, dst]
# for i in range(2):
#     plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'),
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()



# # 读取原始图像
# img = cv2.imread('lena.png')
# # 获取图像高度和宽度
# height = img.shape[0]
# width = img.shape[1]
# # 采样转换成16*16区域
# numHeight = height / 16
# numwidth = width / 16
# # 创建一幅图像
# new_img = np.zeros((height, width, 3), np.uint8)
# # 图像循环采样16*16区域
# for i in range(16):
#     # 获取Y坐标
#     y = i * numHeight
#     for j in range(16):
#         # 获取X坐标
#         x = j * numwidth
#         # 获取填充颜色 左上角像素点
#         b = img[y, x][0]
#         g = img[y, x][1]
#         r = img[y, x][2]
#         # 循环设置小区域采样
#         for n in range(numHeight):
#             for m in range(numwidth):
#                 new_img[y + n, x + m][0] = np.uint8(b)
#                 new_img[y + n, x + m][1] = np.uint8(g)
#                 new_img[y + n, x + m][2] = np.uint8(r)
#             # 显示图像
# cv2.imshow("src", img)
# cv2.imshow("", new_img)
# # 等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 读取原始图像
# img = cv2.imread('lena.png')
# # 获取图像高度和宽度
# height = img.shape[0]
# width = img.shape[1]
# # 采样转换成8*8区域
# numHeight = height / 8
# numwidth = width / 8
# # 创建一幅图像
# new_img = np.zeros((height, width, 3), np.uint8)
# # 图像循环采样8*8区域
# for i in range(8):
#     # 获取Y坐标
#     y = i * numHeight
#     for j in range(8):
#         # 获取X坐标
#         x = j * numwidth
#         # 获取填充颜色 左上角像素点
#         b = img[y, x][0]
#         g = img[y, x][1]
#         r = img[y, x][2]
#         # 循环设置小区域采样
#         for n in range(numHeight):
#             for m in range(numwidth):
#                 new_img[y + n, x + m][0] = np.uint8(b)
#                 new_img[y + n, x + m][1] = np.uint8(g)
#                 new_img[y + n, x + m][2] = np.uint8(r)
#             # 显示图像
# cv2.imshow("src", img)
# cv2.imshow("Sampling", new_img)
# # 等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 读取原始图像
im = cv2.imread('people.png', 1)
# 设置鼠标左键开启
en = False
# 鼠标事件
def draw(event, x, y, flags, param):
    global en
    # 鼠标左键按下开启en值
    if event == cv2.EVENT_LBUTTONDOWN:
        en = True
    # 鼠标左键按下并且移动
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_LBUTTONDOWN:
        # 调用函数打马赛克
        if en:
            drawMask(y, x)
        # 鼠标左键弹起结束操作
        elif event == cv2.EVENT_LBUTTONUP:
            en = False
# 图像局部采样操作
def drawMask(x, y, size=10):
    # size*size采样处理
    m = x / size * size
    n = y / size * size
    print(m, n)
    # 10*10区域设置为同一像素值
    for i in range(size):
        for j in range(size):
            im[m + i][n + j] = im[m][n]
# 打开对话框
cv2.namedWindow('image')
# 调用draw函数设置鼠标操作
cv2.setMouseCallback('image', draw)
# 循环处理
while (1):
    cv2.imshow('image', im)
    # 按ESC键退出
    if cv2.waitKey(10) & 0xFF == 27:
        break
    # 按s键保存图片
    elif cv2.waitKey(10) & 0xFF == 115:
        cv2.imwrite('sava.png', im)
# 退出窗口
cv2.destroyAllWindows()