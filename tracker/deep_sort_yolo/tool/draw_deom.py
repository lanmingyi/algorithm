# cv2解决绘制中文乱码
import cv2
import numpy
from PIL import Image, ImageDraw, ImageFont


def cv2ImgAddText(img, text, left, top, text_color=(0, 255, 0), text_size=20):
    if isinstance(img, numpy.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    font_style = ImageFont.truetype(
        "font/simsun.ttc", text_size, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, text_color, font=font_style)
    # 转换回OpenCV格式
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    imgAddText = cv2ImgAddText(cv2.imread('test.jpeg'), "大家好，cv2解决绘制中文乱码", 10, 65, (0, 0, 139), 20)
    cv2.imshow('show', imgAddText)
    if cv2.waitKey(100000) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
