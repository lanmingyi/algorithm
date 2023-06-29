#encoding:utf-8
import cv2  
import numpy as np  
import matplotlib.pyplot as plt

#读取图片
src = cv2.imread('miao.jpg')
#灰度图像处理
GrayImage = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
#二进制阈值化处理
r, b = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_BINARY)
print (r)
#显示图像
cv2.imshow("src", src)
cv2.imshow("result", b)
#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()


# #读取图片
# src = cv2.imread('miao.jpg')
# #灰度图像处理
# GrayImage = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
# #反二进制阈值化处理
# r, b = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_BINARY_INV)
# print (r)
# #显示图像
# cv2.imshow("src", src)
# cv2.imshow("result", b)
# #等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# #读取图片
# src = cv2.imread('miao.jpg')
# #灰度图像处理
# GrayImage = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
# #截断阈值化处理
# r, b = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_TRUNC)
# print (r)
# #显示图像
# cv2.imshow("src", src)
# cv2.imshow("result", b)
# #等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# #读取图片
# src = cv2.imread('miao.jpg')
# #灰度图像处理
# GrayImage = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
# #反阈值化为0处理
# r, b = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_TOZERO_INV)
# print (r)
# #显示图像
# cv2.imshow("src", src)
# cv2.imshow("result", b)
# #等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# #读取图片
# src = cv2.imread('miao.jpg')
# #灰度图像处理
# GrayImage = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
# #阈值化为0处理
# r, b = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_TOZERO)
# print (r)
# #显示图像
# cv2.imshow("src", src)
# cv2.imshow("result", b)
# #等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# #读取图像
# img=cv2.imread('miao.jpg')
# lenna_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
# #阈值化处理
# ret,thresh1=cv2.threshold(GrayImage,127,255,cv2.THRESH_BINARY)  
# ret,thresh2=cv2.threshold(GrayImage,127,255,cv2.THRESH_BINARY_INV)  
# ret,thresh3=cv2.threshold(GrayImage,127,255,cv2.THRESH_TRUNC)  
# ret,thresh4=cv2.threshold(GrayImage,127,255,cv2.THRESH_TOZERO)  
# ret,thresh5=cv2.threshold(GrayImage,127,255,cv2.THRESH_TOZERO_INV)
# #显示结果
# titles = ['Gray Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']  
# images = [GrayImage, thresh1, thresh2, thresh3, thresh4, thresh5]  
# for i in range(6):  
#    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')  
#    plt.title(titles[i])  
#    plt.xticks([]),plt.yticks([])  
# plt.show()

