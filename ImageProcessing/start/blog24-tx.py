# coding:utf-8
import cv2
import numpy as np

# # 读取原始图像
# src = cv2.imread('scenery.png')
# # 新建目标图像
# dst = np.zeros_like(src)
# # 获取图像行和列
# rows, cols = src.shape[:2]
# # 定义偏移量和随机数
# offsets = 5
# random_num = 0
# # 毛玻璃效果: 像素点邻域内随机像素点的颜色替代当前像素点的颜色
# for y in range(rows - offsets):
#     for x in range(cols - offsets):
#         random_num = np.random.randint(0, offsets)
#         dst[y, x] = src[y + random_num, x + random_num]
# # 显示图像
# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 读取原始图像
img = cv2.imread('../blog24-tx/scenery.png', 1)
# 获取图像的高度和宽度
height, width = img.shape[:2]
# 图像灰度处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 创建目标图像
dstImg = np.zeros((height, width, 1), np.uint8)
# 浮雕特效算法：newPixel = grayCurrentPixel - grayNextPixel + 150
for i in range(0, height):
    for j in range(0, width - 1):
        grayCurrentPixel = int(gray[i, j])
        grayNextPixel = int(gray[i, j + 1])
        newPixel = grayCurrentPixel - grayNextPixel + 150
        if newPixel > 255:
            newPixel = 255
        if newPixel < 0:
            newPixel = 0
        dstImg[i, j] = newPixel
    # 显示图像
cv2.imshow('src', img)
cv2.imshow('dst', dstImg)
# 等待显示
cv2.waitKey()
cv2.destroyAllWindows()


# 读取原始图像
src = cv2.imread('../blog24-tx/scenery.png')
# 图像灰度处理
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# 自定义卷积核
kernel = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])
# 图像浮雕效果
output = cv2.filter2D(gray, -1, kernel)
# 显示图像
cv2.imshow('Original Image', src)
cv2.imshow('Emboss_1', output)
# 等待显示
cv2.waitKey()
cv2.destroyAllWindows()