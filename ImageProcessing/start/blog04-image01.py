# -*- coding:utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取图片
img = cv2.imread("test.jpg", cv2.IMREAD_UNCHANGED)
rows, cols, chn = img.shape
#加噪声
for i in range(5000):    
    x = np.random.randint(0, rows) 
    y = np.random.randint(0, cols)    
    img[x,y,:] = 255
cv2.imshow("noise", img)          
#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()


# #读取图片
# img = cv2.imread('test01.jpg')
# source = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# #均值滤波
# result = cv2.blur(source, (5,5))
# #显示图形
# titles = ['Source Image', 'Blur Image']  
# images = [source, result]  
# for i in range(2):  
#    plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray')  
#    plt.title(titles[i])  
#    plt.xticks([]),plt.yticks([])  
# plt.show()  


# #读取图片
# img = cv2.imread('test01.jpg')
# source = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# #方框滤波
# result = cv2.boxFilter(source, -1, (5,5), normalize=1)
# #显示图形
# titles = ['Source Image', 'BoxFilter Image']  
# images = [source, result]  
# for i in range(2):  
#    plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray')  
#    plt.title(titles[i])  
#    plt.xticks([]),plt.yticks([])  
# plt.show()


# #读取图片
# img = cv2.imread('test01.jpg')
# source = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# #方框滤波
# result = cv2.boxFilter(source, -1, (5,5), normalize=0)
# #显示图形
# titles = ['Source Image', 'BoxFilter Image']  
# images = [source, result]  
# for i in range(2):  
#    plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray')  
#    plt.title(titles[i])  
#    plt.xticks([]),plt.yticks([])  
# plt.show() 


# #读取图片
# img = cv2.imread('test01.jpg')
# source = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# #高斯滤波
# result = cv2.GaussianBlur(source, (3,3), 0)
# #显示图形
# titles = ['Source Image', 'GaussianBlur Image']  
# images = [source, result]  
# for i in range(2):  
#    plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray')  
#    plt.title(titles[i])  
#    plt.xticks([]),plt.yticks([])  
# plt.show()  


# #读取图片
# img = cv2.imread('test01.jpg')
# #高斯滤波
# result = cv2.medianBlur(img, 3)
# #显示图像
# cv2.imshow("source img", img)
# cv2.imshow("medianBlur", result)
# #等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()
