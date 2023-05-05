#encoding:utf-8
import cv2  
import numpy as np  
import matplotlib.pyplot as plt
 
#读取图片
img = cv2.imread('lena-hd.png')
test = img
#方法一：Numpy加法运算
result1 = img + test
#方法二：OpenCV加法运算
result2 = cv2.add(img, test)
#显示图像
cv2.imshow("original", img)
cv2.imshow("result1", result1)
cv2.imshow("result2", result2)
#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()


# #读取图片
# src1 = cv2.imread('test.jpg')
# src2 = cv2.imread('lena-hd.png')
# print (src1.shape)
# print (src2.shape)
# #图像融合
# result = cv2.addWeighted(src1, 1, src2, 1, 0)
# #显示图像
# cv2.imshow("src1", src1)
# cv2.imshow("src2", src2)
# cv2.imshow("result", result)
# #等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# #读取图片
# src = cv2.imread('Lena.png')
# #图像类型转换
# result = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# #显示图像
# cv2.imshow("src", src)
# cv2.imshow("result", result)
# #等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()
