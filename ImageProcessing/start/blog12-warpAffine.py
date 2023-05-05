# encoding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

# # 读取图片
# src = cv2.imread('flower.png')
# # 获取图像大小
# rows, cols = src.shape[:2]
# # 设置图像仿射变换矩阵
# pos1 = np.float32([[50, 50], [200, 50], [50, 200]])
# pos2 = np.float32([[10, 100], [200, 50], [100, 250]])
# M = cv2.getAffineTransform(pos1, pos2)
# # 图像仿射变换
# result = cv2.warpAffine(src, M, (cols, rows))
# # 显示图像
# cv2.imshow("original", src)
# cv2.imshow("result", result)
# # 等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# #读取图片
# src = cv2.imread('flower.png')
# #获取图像大小
# rows, cols = src.shape[:2]
# #设置图像透视变换矩阵
# pos1 = np.float32([[114, 82], [287, 156], [8, 322], [216, 333]])
# pos2 = np.float32([[0, 0], [188, 0], [0, 262], [188, 262]])
# M = cv2.getPerspectiveTransform(pos1, pos2)
# #图像透视变换
# result = cv2.warpPerspective(src, M, (190, 272))
# #显示图像
# cv2.imshow("original", src)
# cv2.imshow("result", result)
# #等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# #读取图片
# src = cv2.imread('test01.jpg')
# #获取图像大小
# rows, cols = src.shape[:2]
# #将源图像高斯模糊
# img = cv2.GaussianBlur(src, (3,3), 0)
# #进行灰度化处理
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# #边缘检测（检测出图像的边缘信息）
# edges = cv2.Canny(gray,50,250,apertureSize = 3)
# cv2.imwrite("canny.jpg", edges)
# #通过霍夫变换得到A4纸边缘
# lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength=90,maxLineGap=10)
# #下面输出的四个点分别为四个顶点
# for x1,y1,x2,y2 in lines[0]:
#     print(x1,y1),(x2,y2)
# for x1,y1,x2,y2 in lines[1]:
#     print(x1,y1),(x2,y2)
# #绘制边缘
# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(gray, (x1,y1), (x2,y2), (0,0,255), 1)
# #根据四个顶点设置图像透视变换矩阵
# pos1 = np.float32([[114, 82], [287, 156], [8, 322], [216, 333]])
# pos2 = np.float32([[0, 0], [188, 0], [0, 262], [188, 262]])
# M = cv2.getPerspectiveTransform(pos1, pos2)
# #图像透视变换
# result = cv2.warpPerspective(src, M, (190, 272))
# #显示图像
# cv2.imshow("original", src)
# cv2.imshow("result", result)
# #等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 读取图片
# img = cv2.imread('Lena.png')
# image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # 图像平移矩阵
# M = np.float32([[1, 0, 80], [0, 1, 30]])
# rows, cols = image.shape[:2]
# img1 = cv2.warpAffine(image, M, (cols, rows))
# # 图像缩小
# img2 = cv2.resize(image, (200, 100))
# # 图像放大
# img3 = cv2.resize(image, None, fx=1.1, fy=1.1)
# # 绕图像的中心旋转
# # 源图像的高、宽 以及通道数
# rows, cols, channel = image.shape
# # 函数参数：旋转中心 旋转度数 scale
# M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
# # 函数参数：原始图像 旋转参数 元素图像宽高
# img4 = cv2.warpAffine(image, M, (cols, rows))
# # 图像翻转
# img5 = cv2.flip(image, 0)  # 参数=0以X轴为对称轴翻转
# img6 = cv2.flip(image, 1)  # 参数>0以Y轴为对称轴翻转
# # 图像的仿射
# pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
# pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
# M = cv2.getAffineTransform(pts1, pts2)
# img7 = cv2.warpAffine(image, M, (rows, cols))
# # 图像的透射
# pts1 = np.float32([[56, 65], [238, 52], [28, 237], [239, 240]])
# pts2 = np.float32([[0, 0], [200, 0], [0, 200], [200, 200]])
# M = cv2.getPerspectiveTransform(pts1, pts2)
# img8 = cv2.warpPerspective(image, M, (200, 200))
# # 循环显示图形
# titles = ['source', 'shift', 'reduction', 'enlarge', 'rotation', 'flipX', 'flipY', 'affine', 'transmission']
# images = [image, img1, img2, img3, img4, img5, img6, img7, img8]
# for i in range(9):
#     plt.subplot(3, 3, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

