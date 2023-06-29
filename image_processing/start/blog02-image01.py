# -*- coding:utf-8 -*-
import cv2
import numpy

#读取图片
img = cv2.imread("picture.bmp", cv2.IMREAD_UNCHANGED)

#灰度图像
p = img[88, 142]
print(p)

#显示图像
cv2.imshow("Demo", img)

#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()

#写入图像
cv2.imwrite("testyxz.jpg", img)


# #读取图片
# img = cv2.imread("test.jpg", cv2.IMREAD_UNCHANGED)

# #Numpy读取像素
# print(img.item(78, 100, 0))
# print(img.item(78, 100, 1))
# print(img.item(78, 100, 2))
# img.itemset((78, 100, 0), 100)
# img.itemset((78, 100, 1), 100)
# img.itemset((78, 100, 2), 100)
# print(img.item(78, 100, 0))
# print(img.item(78, 100, 1))
# print(img.item(78, 100, 2))

# #获取图像形状
# print(img.shape)
# #获取像素数目
# print(img.size)
# #获取图像类型
# print(img.dtype)


