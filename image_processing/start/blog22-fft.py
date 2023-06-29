# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import cv2
from matplotlib import pyplot as plt

# # 读取图像
# img = cv.imread('lena.png', 0)
# # 快速傅里叶变换算法得到频率分布
# f = np.fft.fft2(img)
# # 默认结果中心点位置是在左上角,
# # 调用fftshift()函数转移到中间位置
# fshift = np.fft.fftshift(f)
# # fft结果是复数, 其绝对值结果是振幅
# fimg = np.log(np.abs(fshift))
# # 展示结果
# plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Fourier')
# plt.axis('off')
# plt.subplot(122), plt.imshow(fimg, 'gray'), plt.title('Fourier Fourier')
# plt.axis('off')
# plt.show()


# # 读取图像
# img = cv.imread('lena.png', 0)
# # 傅里叶变换
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# res = np.log(np.abs(fshift))
# # 傅里叶逆变换
# ishift = np.fft.ifftshift(fshift)
# iimg = np.fft.ifft2(ishift)
# iimg = np.abs(iimg)
# # 展示结果
# plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
# plt.axis('off')
# plt.subplot(132), plt.imshow(res, 'gray'), plt.title('Fourier Image')
# plt.axis('off')
# plt.subplot(133), plt.imshow(iimg, 'gray'), plt.title('Inverse Fourier Image')
# plt.axis('off')
# plt.show()


# # 读取图像
# img = cv2.imread('lena.png', 0)
# # 傅里叶变换
# dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
# # 将频谱低频从左上角移动至中心位置
# dft_shift = np.fft.fftshift(dft)
# # 频谱图像双通道复数转换为0-255区间
# result = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
# # 显示图像
# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(result, cmap='gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()


# 读取图像
img = cv2.imread('../blog22-fft/lena.png', 0)
# 傅里叶变换
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dftshift = np.fft.fftshift(dft)
res1 = 20 * np.log(cv2.magnitude(dftshift[:, :, 0], dftshift[:, :, 1]))
# 傅里叶逆变换
ishift = np.fft.ifftshift(dftshift)
iimg = cv2.idft(ishift)
res2 = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
# 显示图像
plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(132), plt.imshow(res1, 'gray'), plt.title('Fourier Image')
plt.axis('off')
plt.subplot(133), plt.imshow(res2, 'gray'), plt.title('Inverse Fourier Image')
plt.axis('off')
plt.show()