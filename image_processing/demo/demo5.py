# coding=utf-8
# author:兰明易 time:2020-03-09
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# img = Image.open('3.jpg')
# im1=img.rotate(45)
# im1.show()

im = Image.open("3.jpg")
im_45 = im.rotate(45)
im_30 = im.rotate(30, Image.NEAREST, 1)
print(im_45.size,im_30.size)
im_45.show()
# im_30.show()


# img.save('test.tiff')  # 保存图像
# img.save('lena.jpg')
# img.save('lena.bmp')

# img1 = Image.open('lena.jpg')
# img2 = Image.open('lena.bmp')

# print('image:', img.format)  # 图像格式
# # print('size:', img.size)  # 图像尺寸
# # print('mode:', img.mode)  # 色彩模式
# # print('image1:', img1.format)
# # print('image2:', img2.format)
# plt.figure(figsize=(5, 5))
# plt.imshow(img)  # 对图像进行处理，并显示图像格式
# plt.show()  # 显示图像

# plt.figure(figsize=(15, 5))  # 设置画布尺寸
# plt.subplot(131)  # 将画布划分成1行3列
# plt.axis("off")  # 不显示坐标轴
# plt.imshow(img)
# plt.title(img.format)
#
# plt.subplot(132)
# plt.axis("off")
# plt.imshow(img1)
# plt.title(img1.format)
#
# plt.subplot(133)
# plt.axis("off")
# plt.imshow(img2)
# plt.title(img2.format)
# plt.show()

# img_gray = img.convert("L")  # 转化为灰度图像
# print("mode:", img_gray.mode)
# plt.figure(figsize=(5, 5))
# plt.imshow(img_gray)
# plt.show()

# img_r, img_g, img_b = img.split()  # 颜色分离
# plt.figure(figsize=(10, 10))
# plt.subplot(221)
# plt.axis("off")
# plt.imshow(img_r, cmap="gray")
# plt.title("R", fontsize=20)
#
# plt.subplot(222)
# plt.axis("off")
# plt.imshow(img_g, cmap="gray")
# plt.title("G", fontsize=20)
#
# plt.subplot(223)
# plt.axis("off")
# plt.imshow(img_b, cmap="gray")
# plt.title("B", fontsize=20)
#
# img_rgb = Image.merge("RGB", [img_r, img_g, img_b])  # 颜色合并
# plt.subplot(2, 2, 4)
# plt.axis("off")
# plt.imshow(img_rgb)
# plt.title("RGB", fontsize=20)
# plt.show()

# arr_img = np.array(img)  # 转化为数组
# print("shape:", arr_img.shape, "\n")
# print(arr_img)

# arr_img_new = 255 - arr_img  # 对图像中每一个像素做反色处理
# plt.figure(figsize=(10, 5))
# plt.subplot(121)
# plt.axis("off")
# plt.imshow(arr_img, cmap='gray')
# plt.subplot(122)
# plt.axis("off")
# plt.imshow(arr_img_new, cmap='gray')
# plt.show()

# plt.figure(figsize=(5, 5))
# img_small = img.resize((64, 64))  # 缩放图像
# img_small.save("lna_s.jpg")
# plt.imshow(img_small)
# plt.show()

# plt.rcParams["font.sans-serif"] = "SimHei"  # 设置中文字体, run configuration Params运行配置参数，SimHei中文黑体
# # plt.rcdefaults()  # 恢复标准默认配置
# img_flr = img.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转
# img_r90 = img.transpose(Image.ROTATE_270)  # 逆时针旋转90度
# img_r90.save('4-5.jpg')
# img_tp = img.transpose(Image.TRANSPOSE)  # 转置
# plt.figure(figsize=(5, 5))
# plt.subplot(221)
# plt.axis('off')
# plt.imshow(img_flr)
# plt.title('左右旋转', fontsize=20)
#
# plt.subplot(222)
# plt.axis('off')
# plt.imshow(img_r90)
#
# plt.subplot(223)
# plt.axis('off')
# plt.imshow(img_tp)
#
# plt.subplot(224)
# plt.axis('off')
# plt.imshow(img)
# plt.title("原图")
# plt.show()

# # 裁剪
# img_region = img.crop((100, 100, 400, 400))  # 100,100左上角；400，400右下角
# plt.imshow(img_region)
# plt.show()

