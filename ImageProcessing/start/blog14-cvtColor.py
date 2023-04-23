# encoding:utf-8
import cv2
import numpy as np

# # 读取原始图片
# src = cv2.imread('miao.jpg')
# # 图像灰度化处理
# grayImage = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# # 显示图像
# cv2.imshow("src", src)
# cv2.imshow("result", grayImage)
# # 等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 读取原始图像
# img_BGR = cv2.imread('miao.jpg')
# # BGR转换为RGB
# img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
# # 灰度化处理
# img_GRAY = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
# # BGR转HSV
# img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
# # BGR转YCrCb
# img_YCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
# # BGR转HLS
# img_HLS = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HLS)
# # BGR转XYZ
# img_XYZ = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2XYZ)
# # BGR转LAB
# img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
# # BGR转YUV
# img_YUV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YUV)
# # 调用matplotlib显示处理结果
# titles = ['BGR', 'RGB', 'GRAY', 'HSV', 'YCrCb', 'HLS', 'XYZ', 'LAB', 'YUV']
# images = [img_BGR, img_RGB, img_GRAY, img_HSV, img_YCrCb,
#           img_HLS, img_XYZ, img_LAB, img_YUV]
# for i in range(9):
#     plt.subplot(3, 3, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()


# # 读取原始图像
# img = cv2.imread('miao.jpg')
# # 获取图像高度和宽度
# height = img.shape[0]
# width = img.shape[1]
# # 创建一幅图像
# grayimg = np.zeros((height, width, 3), np.uint8)
# # 图像最大值灰度处理
# for i in range(height):
#     for j in range(width):
#         # 获取图像R G B最大值
#         gray = max(img[i, j][0], img[i, j][1], img[i, j][2])
#         # 灰度图像素赋值 gray=max(R,G,B)
#         grayimg[i, j] = np.uint8(gray)
#
# # 显示图像
# cv2.imshow("src", img)
# cv2.imshow("gray", grayimg)
# # 等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 读取原始图像
# img = cv2.imread('miao.jpg')
# # 获取图像高度和宽度
# height = img.shape[0]
# width = img.shape[1]
# # 创建一幅图像
# grayimg = np.zeros((height, width, 3), np.uint8)
# print(grayimg)
# # 图像平均灰度处理方法
# for i in range(height):
#     for j in range(width):
#         # 灰度值为RGB三个分量的平均值
#         gray = (int(img[i, j][0]) + int(img[i, j][1]) + int(img[i, j][2])) / 3
#         grayimg[i, j] = np.uint8(gray)
# # 显示图像
# cv2.imshow("src", img)
# cv2.imshow("gray", grayimg)
# # 等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 读取原始图像
# img = cv2.imread('miao.jpg')
# # 获取图像高度和宽度
# height = img.shape[0]
# width = img.shape[1]
# # 创建一幅图像
# grayimg = np.zeros((height, width, 3), np.uint8)
# print(grayimg)
# # 图像平均灰度处理方法
# for i in range(height):
#     for j in range(width):
#         # 灰度加权平均法
#         gray = 0.30 * img[i, j][0] + 0.59 * img[i, j][1] + 0.11 * img[i, j][2]
#         grayimg[i, j] = np.uint8(gray)
#
# # 显示图像
# cv2.imshow("src", img)
# cv2.imshow("gray", grayimg)
# # 等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()