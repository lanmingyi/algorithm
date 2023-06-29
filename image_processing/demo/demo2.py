# coding=utf-8
# author:兰明易 time:2020-05-03
import numpy as np
from PIL import Image
# im = Image.open('3.png').convert('RGB')
# arr = np.array(im)


def blackWithe(imagename):  # 彩色转黑白
    # r,g,b = r*0.299+g*0.587+b*0.114
    im = np.asarray(Image.open(imagename).convert('RGB'))
    trans = np.array([[0.299,0.587,0.114],[0.299,0.587,0.114],[0.299,0.587,0.114]]).transpose()
    im = np.dot(im,trans)
    return Image.fromarray(np.array(im).astype('uint8'))


def fleeting(imagename,params=12):  # 流年
    im = np.asarray(Image.open(imagename).convert('RGB'))
    im1 = np.sqrt(im*[1.0,0.0,0.0])*params
    im2 = im*[0.0,1.0,1.0]
    im = im1+im2
    return Image.fromarray(np.array(im).astype('uint8'))


def oldFilm(imagename):  # 旧电影
    im = np.asarray(Image.open(imagename).convert('RGB'))
    # r=r*0.393+g*0.769+b*0.189 g=r*0.349+g*0.686+b*0.168 b=r*0.272+g*0.534b*0.131
    trans = np.array([[0.393,0.769,0.189],[0.349,0.686,0.168],[0.272,0.534,0.131]]).transpose()
    # clip 超过255的颜色置为255
    im = np.dot(im,trans).clip(max=255)
    return Image.fromarray(np.array(im).astype('uint8'))


def reverse(imagename):  # 反色
    im = 255 - np.asarray(Image.open(imagename).convert('RGB'))
    return Image.fromarray(np.array(im).astype('uint8'))


# image = blackWithe('3.jpg')
# image = fleeting('3.jpg')
# image = oldFilm('3.jpg')
image = reverse('3.jpg')
# image.save('2-2-1.jpg')
image.show()
