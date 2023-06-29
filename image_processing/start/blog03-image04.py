# -*- coding:utf-8 -*-
import cv2
import numpy as np

#读取图片
img = cv2.imread("test.jpg", cv2.IMREAD_UNCHANGED)

#定义200*100矩阵 3对应BGR
face = np.ones((200, 100, 3))

#显示原始图像
cv2.imshow("Demo", img)

#显示ROI区域
face = img[200:400, 200:300]
cv2.imshow("face", face)

#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()


# #读取图片
# img = cv2.imread("test.jpg", cv2.IMREAD_UNCHANGED)
# #定义300*100矩阵 3对应BGR
# face = np.ones((200, 200, 3))
# #显示原始图像
# cv2.imshow("Demo", img)
# #显示ROI区域
# face = img[100:300, 250:450]
# img[0:200,0:200] = face
# cv2.imshow("face", img)


# #等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# #读取图片
# img = cv2.imread("test.jpg", cv2.IMREAD_UNCHANGED)
# test = cv2.imread("test3.jpg", cv2.IMREAD_UNCHANGED)
# print (test.shape)
# #定义300*100矩阵 3对应BGR
# face = np.ones((200, 200))
# #显示原始图像
# cv2.imshow("Demo", img)
# #显示ROI区域
# face = img[100:300, 250:450]
# test[200:400, 200:400] = face
# cv2.imshow("Pic", test)
# #等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# #读取图片
# img = cv2.imread("test.jpg", cv2.IMREAD_UNCHANGED)
# #拆分通道
# b, g, r = cv2.split(img)
# #显示原始图像
# cv2.imshow("B", b)
# cv2.imshow("G", g)
# cv2.imshow("R", r)     
# #等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #读取图片
# img = cv2.imread("test.jpg", cv2.IMREAD_UNCHANGED)
# #拆分通道
# b, g, r = cv2.split(img)
# #合并通道
# m = cv2.merge([b, g, r])
# cv2.imshow("Merge", m)
# #等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #读取图片
# img = cv2.imread("test.jpg", cv2.IMREAD_UNCHANGED)
# rows, cols, chn = img.shape
# #拆分通道
# b = cv2.split(img)[0]
# g = np.zeros((rows,cols),dtype=img.dtype)
# r = np.zeros((rows,cols),dtype=img.dtype)
# #合并通道
# m = cv2.merge([b, g, r])
# cv2.imshow("Merge", m)    
# #等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()



