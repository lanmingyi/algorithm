# coding:utf-8
import cv2
import math
import numpy as np

# # 读取原始图像
# img = cv2.imread('scenery.png')
# # 图像灰度处理
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # 高斯滤波降噪
# gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
# # Canny算子
# canny = cv2.Canny(gaussian, 50, 150)
# # 阈值化处理
# ret, result = cv2.threshold(canny, 100, 255, cv2.THRESH_BINARY_INV)
# # 显示图像
# cv2.imshow('src', img)
# cv2.imshow('result', result)
# cv2.waitKey()
# cv2.destroyAllWindows()


# # 读取原始图像
# img = cv2.imread('scenery.png')
# # 获取图像行和列
# rows, cols = img.shape[:2]
# # 新建目标图像
# dst = np.zeros((rows, cols, 3), dtype="uint8")
# # 图像怀旧特效
# for i in range(rows):
#     for j in range(cols):
#         B = 0.272 * img[i, j][2] + 0.534 * img[i, j][1] + 0.131 * img[i, j][0]
#         G = 0.349 * img[i, j][2] + 0.686 * img[i, j][1] + 0.168 * img[i, j][0]
#         R = 0.393 * img[i, j][2] + 0.769 * img[i, j][1] + 0.189 * img[i, j][0]
#         if B > 255:
#             B = 255
#         if G > 255:
#             G = 255
#         if R > 255:
#             R = 255
#         dst[i, j] = np.uint8((B, G, R))
# # 显示图像
# cv2.imshow('src', img)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


# # 读取原始图像
# img = cv2.imread('scenery.png')
# # 获取图像行和列
# rows, cols = img.shape[:2]
# # 设置中心点
# centerX = rows / 2
# centerY = cols / 2
# print(centerX, centerY)
# radius = min(centerX, centerY)
# print(radius)
# # 设置光照强度
# strength = 200
# # 新建目标图像
# dst = np.zeros((rows, cols, 3), dtype="uint8")
# # 图像光照特效
# for i in range(rows):
#     for j in range(cols):
#         # 计算当前点到光照中心距离(平面坐标系中两点之间的距离)
#         distance = math.pow((centerY - j), 2) + math.pow((centerX - i), 2)
#         # 获取原始图像
#         B = img[i, j][0]
#         G = img[i, j][1]
#         R = img[i, j][2]
#         if (distance < radius * radius):
#             # 按照距离大小计算增强的光照值
#             result = (int)(strength * (1.0 - math.sqrt(distance) / radius))
#             B = img[i, j][0] + result
#             G = img[i, j][1] + result
#             R = img[i, j][2] + result
#             # 判断边界 防止越界
#             B = min(255, max(0, B))
#             G = min(255, max(0, G))
#             R = min(255, max(0, R))
#             dst[i, j] = np.uint8((B, G, R))
#         else:
#             dst[i, j] = np.uint8((B, G, R))
# # 显示图像
# cv2.imshow('src', img)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


# # 读取原始图像
# img = cv2.imread('scenery.png')
# # 获取图像行和列
# rows, cols = img.shape[:2]
# # 新建目标图像
# dst = np.zeros((rows, cols, 3), dtype="uint8")
# # 图像流年特效
# for i in range(rows):
#     for j in range(cols):
#         # B通道的数值开平方乘以参数12
#         B = math.sqrt(img[i, j][0]) * 12
#         G = img[i, j][1]
#         R = img[i, j][2]
#         if B > 255:
#             B = 255
#         dst[i, j] = np.uint8((B, G, R))
# # 显示图像
# cv2.imshow('src', img)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 获取滤镜颜色
def getBGR(img, table, i, j):
    # 获取图像颜色
    b, g, r = img[i][j]
    # 计算标准颜色表中颜色的位置坐标
    x = int(g / 4 + int(b / 32) * 64)
    y = int(r / 4 + int((b % 32) / 4) * 64)
    # 返回滤镜颜色表中对应的颜色
    return lj_map[x][y]
# 读取原始图像
img = cv2.imread('../blog25-lvjing/scenery.png')
lj_map = cv2.imread('table.png')
print(img.shape)
# 获取图像行和列
rows, cols = img.shape[:2]
# 新建目标图像
dst = np.zeros((rows, cols, 3), dtype="uint8")
# 循环设置滤镜颜色
for i in range(rows):
    for j in range(cols):
        dst[i][j] = getBGR(img, lj_map, i, j)
# 显示图像
cv2.imshow('src', img)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()