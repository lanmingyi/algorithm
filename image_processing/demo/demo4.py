# coding=utf-8
# author:兰明易 time:2020-05-03
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img=np.array(Image.open('1.jpg'))  # 随机生成5000个椒盐
rows, cols, dims = img.shape
for i in range(50000):
    x = np.random.randint(0, rows)
    y = np.random.randint(0, cols)
    img[x, y, :]=255
plt.figure("beauty")
plt.imshow(img)
plt.axis('off')
# plt.show()
plt.savefig('1-4-1.jpg')

