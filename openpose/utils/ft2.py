"""
Copyright (c) 2021 Bright Mind. All rights reserved.
Written by MingYi Lan

http://blog.csdn.net/zizi7/article/details/70145150
"""

import numpy as np
import freetype
import copy


class PutChineseText(object):
    def __init__(self, ttf):
        self._face = freetype.Face(ttf)

    def draw_text(self, image, pos, text, text_size, text_color):
        """
        draw chinese(or not) text with ttf
        :param image:     image(numpy.ndarray) to draw text
        :param pos:       where to draw text
        :param text:      the context, for chinese should be unicode type
        :param text_size: text size
        :param text_color:text color
        :return:          image
        """
        self._face.set_char_size(text_size * 64)
        metrics = self._face.size
        ascender = metrics.ascender / 64.0

        # descender = metrics.descender / 64.0
        # height = metrics.height / 64.0
        # linegap = height - ascender + descender
        ypos = int(ascender)

        if not isinstance(text, str):
            # 对于Python 2中的unicode和Python 3中的str，对于Python 2中的str/bytes和Python 3中的bytes的二进制文件
            text = text.decode("utf-8")
        img = self.draw_string(image, pos[0], pos[1] + ypos, text, text_color)
        return img

    def draw_string(self, img, x_pos, y_pos, text, color):
        """
        draw string
        :param x_pos: text x-postion on img
        :param y_pos: text y-postion on img
        :param text:  text (unicode)
        :param color: text color
        :return:      image
        """
        prev_char = 0
        pen = freetype.Vector()
        pen.x = x_pos << 6  # div 64，移位操作左移
        pen.y = y_pos << 6

        hscale = 1.0
        matrix = freetype.Matrix(
            int(hscale * 0x10000), int(0.2 * 0x10000), int(0.0 * 0x10000), int(1.1 * 0x10000)
        )
        cur_pen = freetype.Vector()
        pen_translate = freetype.Vector()

        image = copy.deepcopy(img)
        for cur_char in text:
            self._face.set_transform(matrix, pen_translate)
            self._face.load_char(cur_char)
            kerning = self._face.get_kerning(prev_char, cur_char)
            pen.x += kerning.x
            slot = self._face.glyph
            bitmap = slot.bitmap

            cur_pen.x = pen.x
            cur_pen.y = pen.y - slot.bitmap_top * 64
            self.draw_ft_bitmap(image, bitmap, cur_pen, color)

            pen.x += slot.advance.x
            prev_char = cur_char

        return image

    def draw_ft_bitmap(self, img, bitmap, pen, color):
        """
        draw each char
        :param bitmap: bitmap
        :param pen:    pen
        :param color:  pen color e.g.(0,0,255) - red
        :return:       image
        """
        x_pos = pen.x >> 6
        y_pos = pen.y >> 6
        cols = bitmap.width
        rows = bitmap.rows

        glyph_pixels = bitmap.buffer

        for row in range(rows):
            for col in range(cols):
                if glyph_pixels[row * cols + col] != 0:
                    img[y_pos + row][x_pos + col][0] = color[0]
                    img[y_pos + row][x_pos + col][1] = color[1]
                    img[y_pos + row][x_pos + col][2] = color[2]


if __name__ == "__main__":
    # just for test
    import cv2
    id_num = 10

    line = "测试:{}".format(id_num)
    img = np.zeros([300, 300, 3])

    color_ = (0, 0, 255)  # Red
    pos = (3, 3)
    text_size = 24

    # ft = put_chinese_text("wqy-zenhei.ttc")
    ft = PutChineseText("../data/simhei.ttf")
    image = ft.draw_text(img, pos, line, text_size, color_)

    cv2.imshow("display", image)
    cv2.waitKey(0)

