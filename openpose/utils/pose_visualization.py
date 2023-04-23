"""
Copyright (c) 2021 Bright Mind. All rights reserved.
Written by MingYi Lan
"""

import colorsys
import os
import sys
import time

import cv2
import numpy as np
from PIL import ImageFont, Image, ImageDraw

from . import processing
from .ft2 import PutChineseText


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]
    作为整数表示，char是有符号的，数值范围-128～127. uchar是无符号的，数值范围0～255。作为字符表示无区别
    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255 * r), int(255 * g), int(255 * b)


# def create_unique_color_uchar(tag, hue_step=0.41):
#     h, v = (tag * hue_step) % 1, 1.0 - (int(tag * hue_step) % 4) / 5.0
#     r, g, b = colorsys.hsv_to_rgb(h, 1.0, v)
#     return int(255 * r), int(255 * g), int(255 * b)


def draw_skeleton25(color, x, y, centers, frame):
    CocoPairs = [
        (1, 2),
        (1, 5),
        (2, 3),
        (3, 4),
        (5, 6),
        (6, 7),
        (1, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (8, 12),
        (12, 13),
        (13, 14),
        (14, 19),
        (19, 20),
        (14, 21),
        (11, 22),
        (22, 23),
        (11, 24),
        (1, 0),
        (0, 15),
        (15, 17),
        (0, 16),
        (16, 18),
        (2, 17),
        (5, 18),
    ]  # = 19
    # CocoPairsRender = CocoPairs[:-2]
    CocoPairsRender = CocoPairs[:-7]
    show_points = [i for i in range(1, 15)]
    show_points1 = [i for i in range(19, 25)]
    show_points.extend(show_points1)

    if x.count(0) < 17:  # filter out the skeleton with less keypoints
        # draw point
        for i in show_points:
            center = (int(x[i] + 0.5), int(y[i] + 0.5))
            centers[i] = center
            if center == (0, 0):
                continue
            cv2.circle(frame, center, 3, color, thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(CocoPairsRender):
            if centers[pair[0]] == (0, 0) or centers[pair[1]] == (0, 0):
                continue
            cv2.line(frame, centers[pair[0]], centers[pair[1]], color, 2)


def cv2image(cvimg):
    cv2img = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype("data/simhei.ttf", 30, encoding="utf-8")
    # font = ImageFont.truetype("pose/data/simhei.ttf", 30, encoding="utf-8")
    return pilimg, draw, font


def touch_show(touchshowdic, draw, centers, id, new_color, font):
    # print(self.touchshowdic[id])
    # print(self.onlytouch)
    # print(self.nottouch)
    if id in touchshowdic and len(centers) > 0:
        # print(id)
        show_str = touchshowdic[id]
        draw.text((centers[1][0] - 20, centers[1][1] - 20), show_str, new_color, font=font)
        # print(show_str)


def draw_action_label(touchshowdic, centers, id, color, img):
    new_color = (color[2], color[1], color[0])
    pilimg, draw, font = cv2image(img)

    touch_show(touchshowdic, draw, centers, id, new_color, font)
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

    return cv2charimg


def draw_Polygons(image, zones):
    colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [255, 0, 255]]
    if zones == []:
        pass
    else:
        for i in range(0, len(zones)):
            color_idx = i
            zoneLength = len(zones[i])
            zone_list = []

            for j in range(0, zoneLength, 2):
                zone_list.append([zones[i][j], zones[i][j + 1]])
            img_cpy = image.copy()

            cv2.fillConvexPoly(img_cpy, np.array(zone_list), colors[int(color_idx % 5)])
            cv2.addWeighted(image, 0.7, img_cpy, 1 - 0.7, 0.0, image)

    return image


def draw_actions(touchshowdic, img, skeleton, touch_zones):
    # print('img:', img)
    centers = {}
    img_copy = img.copy()
    draw_Polygons(img_copy, touch_zones)
    for p in skeleton.poses:
        keys = p.get_flat_keypoints()
        id = p.id
        x = keys[0::3]
        y = keys[1::3]
        color = create_unique_color_uchar(id)
        draw_skeleton25(color, x, y, centers, img_copy)
        img_copy = draw_action_label(touchshowdic, centers, id, color, img_copy)

    return img_copy


def rectangle(image, xmin, ymin, xmax, ymax, thickness, color=(0, 0, 0), label=None):
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)

    cv2.rectangle(image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=color, thickness=thickness)

    if label is not None:
        # getTextSize: return the size of a box that contains the specified text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 0.4, 1)
        center = pt1[0] + 2, pt1[1] + 6 + text_size[0][1]
        pt2 = pt1[0] + 29 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
        cv2.rectangle(image, pt1, pt2, color, -1)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    pass


def draw_trackers(image, tracker_result):
    """
    tracker_result:
    {‘trackId’ : {‘frameIdx’ :
    {'det_box': [xmin, ymin, xmax, ymax], 'confident': *, 'state': *, 'track_box': [xmin, ymin, xmax, ymax], 'timestamp': '00:00:00', 'det_idx': *, 'update_times': *}
    ......}}
    """

    # state = 2: confirmed
    # update_times: Without detection information, the number of tracking positions is predicted.
    # time_since_update : Total number of frames since last measurement update.
    for track_id, track_info in tracker_result.items():
        for frame_id, track_data in track_info.items():
            # if (int(track_data['state']) != 2:
            if (int(track_data['state'])) != 2 or (int(track_data['update_times'])) > 0:
                continue
            pass

            track_box = track_data['track_box']
            if len(track_box) != 0:
                r, g, b = create_unique_color_uchar(int(track_id))
                xmin = int(track_box[0])
                ymin = int(track_box[1])
                xmax = int(track_box[2])
                ymax = int(track_box[3])
                color = (b, g, r)
                thickness = 2
                label = track_id
                rectangle(image, xmin, ymin, xmax, ymax, thickness, color, label)
            pass
        pass
    pass


def draw_classifications(image, tracker_result, classifications_dict):
    """
    tracker_result:
    {'trackId' : {'frameIdx' :
    {'det_box': [xmin, ymin, xmax, ymax], 'confident': *, 'state': *, 'track_box': [xmin, ymin, xmax, ymax], 'timestamp': '00:00:00', 'det_idx': *, 'update_times': *}
    ......}}
    """
    # state = 2: confirmed
    # update_times: Without detection information, the number of tracking positions is predicted.
    # time_since_update : Total number of frames since last measurement update.
    for track_id, track_info in tracker_result.items():
        for frame_id, track_data in track_info.items():
            if (int(track_data['state'])) != 2 or (int(track_data['update_times'])) > 0:
                continue

            track_box = track_data['track_box']
            if len(track_box) != 0:
                r, g, b = create_unique_color_uchar(int(track_id))
                xmin = int(track_box[0])
                ymin = int(track_box[1])
                xmax = int(track_box[2])
                ymax = int(track_box[3])
                color = (b, g, r)
                thickness = 2
                label = classifications_dict[track_id]
                rectangle(image, xmin, ymin, xmax, ymax, thickness, color, label)
            pass


def draw_detections(image, dets):
    for det in dets:
        xmin = int(det['left'])
        ymin = int(det['top'])
        xmax = int(det['right'])
        ymax = int(det['bottom'])
        category = det['name']

        color = (0, 255, 0)
        cv2.rectangle(image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=color, thickness=2)
        pad_score = "%.2f" % (det['confident'])
        cv2.rectangle(image, pt1=(xmin, ymin), pt2=(xmax, ymin + 16), color=color, thickness=2)
        cv2.putText(image, text=category + ": " + str(pad_score), org=(xmin + 2, ymin + 11), fontFace=0, fontScale=0.4,
                    thickness=1, color=color)
    pass


def draw_detections_time(image, dets, color=(0, 255, 0)):
    # font = ImageFont.truetype("data/simhei.ttf", 16)
    # ==================ft===================
    counting_ft = PutChineseText("data/simhei.ttf")
    text_ft_size = 12

    for det in dets:
        xmin = int(det['left'])
        ymin = int(det['top'])
        xmax = int(det['right'])
        ymax = int(det['bottom'])
        category = det['name']

        cv2.rectangle(image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=color, thickness=2)
        pad_score = "%.2f" % (det['confident'])

        # cv2.rectangle(image, pt1=(xmin, ymin), pt2=(xmax, ymin + 16), color=color, thickness=2)
        # cv2.putText(image, text=category + ": " + str(pad_score), org=(xmin + 2, ymin + 11 ), fontFace=0,
        # fontScale=0.4, thickness=1, color=color)

        # =================chinese==================================
        # image = Image.fromarray(image)
        # draw = ImageDraw.Draw(image)
        # label = ""
        # if category == "person":
        #     label = "员工"
        # if category == "DianJi":
        #     label = "电机"
        # text = label + ":" + str(pad_score)
        # draw.text((int(xmin + 1), int(ymin + 1)), text, tuple(color), font=font)
        # image = np.asarray(image)

        # ===================ft==============================
        if category == "person":
            label = "员工" + ":" + str(pad_score)
        elif category == "DianJi":
            label = "电机" + ":" + str(pad_score)
        elif category == "JieChuPan":
            label = "接触盘" + ":" + str(pad_score)
        else:
            label = "其他"

        if (xmin >= 1820 or ymin > 1000):
            return image

        image = counting_ft.draw_text(image, (int(xmin + 1), int(ymin + 1)), label, text_ft_size, color)
    pass

    return image


def draw_video_info(image, dimensions=None, framerate=None):
    xmin = 8
    ymin = 16
    color = (0, 255, 255)

    if framerate is not None:
        framerate_string = "Framerate: %s frames per second" % (str(framerate))
        text_size = cv2.getTextSize(framerate_string, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(image, text=framerate_string, org=(xmin, ymin), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
                    thickness=2, color=color)
    pass
    if dimensions is not None:
        dimensions_string = "Dimensions: %s x %s" % (str(dimensions[0]), str(dimensions[1]))
        text_size = cv2.getTextSize(dimensions_string, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_size_height = text_size[0][1] + 6
        cv2.putText(image, text=dimensions_string, org=(xmin, ymin + text_size_height),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2, color=color)
    pass


def draw_centroid(image, tracker_result):
    """
    tracker_result:
    {‘trackId’ : {‘frameIdx’ :
    {'det_box': [xmin, ymin, xmax, ymax], 'confident': *, 'state': *, 'track_box': [xmin, ymin, xmax, ymax], 'timestamp': '00:00:00', 'det_idx': *, 'update_times': *}
    ......}}
    """
    for track_id, track_info in tracker_result.items():
        for frame_id, track_data in track_info.items():
            if (int(track_data['state'])) != 2 or (int(track_data['update_times'])) > 0:
                continue
            pass

            track_box = track_data['track_box']
            if len(track_box) != 0:
                r, g, b = create_unique_color_uchar(int(track_id))
                xmin = int(track_box[0])
                ymin = int(track_box[1])
                xmax = int(track_box[2])
                ymax = int(track_box[3])

                center_x = int((xmin + xmax) / 2.0)
                center_y = int((ymin + ymax) / 2.0)

                color = (b, g, r)
                thickness = -1
                radius = 3
                cv2.circle(image, (center_x, center_y), radius, color, thickness)
            pass
        pass
    pass


def draw_centroid_detections(image_cpu, dets, color=(0, 0, 255)):
    for det in dets:
        xmin = int(det['left'])
        ymin = int(det['top'])
        xmax = int(det['right'])
        ymax = int(det['bottom'])

        center_x = int((xmin + xmax) / 2.0)
        center_y = int((ymin + ymax) / 2.0)

        thickness = -1
        radius = 5
        cv2.circle(image_cpu, (center_x, center_y), radius, color, thickness)

    pass


def draw_detections_centroid(image, dets):
    for det in dets:
        xmin = int(det['left'])
        ymin = int(det['top'])
        xmax = int(det['right'])
        ymax = int(det['bottom'])

        center_x = int((xmin + xmax) / 2.0)
        center_y = int((ymin + ymax) / 2.0)

        color = (0, 0, 255)
        thickness = -1
        radius = 3
        cv2.circle(image, (center_x, center_y), radius, color, thickness)


def draw_polylines(image, polylines):
    # polylines = [[polyline_id, x, y, x, y, x, y], ...]
    # --polyline 0.15 0.8014 0.8195 0.1917
    # 0 % 5 = 0, 1 % 5 = 1
    colors = [[255, 0, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]
    color_idx = 0
    for polyline in polylines:
        len_polyline = len(polyline)
        for i in range(1, len_polyline - 2, 2):
            point1 = (int(polyline[i]), int(polyline[i + 1]))
            point2 = (int(polyline[i + 2]), int(polyline[i + 3]))

            cv2.line(image, point1, point2, colors[int(color_idx % 5)], 2)
            thickness = -1
            radius = 4
            cv2.circle(image, point1, radius, colors[int(color_idx % 5)], thickness)
        pass
        color_idx += 1
    pass


def draw_counting(image, polylines, counter_output):
    # polylines = [[polyline_id, x, y, x, y, x, y], ...]
    colors = [[18, 153, 255], [31, 102, 156], [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [255, 0, 255]]
    color_idx = 0
    out_image = image

    for polyline in polylines:
        if len(polyline) < 5:
            continue
        pass

        pt1 = (polyline[1], polyline[2])
        pt2 = (polyline[3], polyline[4])
        id = polyline[0]

        counting_ft = PutChineseText("data/simhei.ttf")

        # label = "enter: " + str(counter_output.get(id).get("enter")) + " exit: " + str(
        #    counter_output.get(id).get("exit"))

        label = "进: " + str(counter_output.get(id).get("enter")) + " 出: " + str(counter_output.get(id).get("exit"))

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)
        left = pt1[0] + (pt2[0] - pt1[0]) / 2 - text_size[0][0] / 2
        top = pt1[1] + (pt2[1] - pt2[0]) / 2 - text_size[0][1] / 2
        # cv2.putText(image, label, (int(left), int(top)), cv2.FONT_HERSHEY_COMPLEX, 0.6, colors[int(color_idx % 5)], 1)
        text_ft_size = 24
        out_image = counting_ft.draw_text(out_image, (int(left), int(top)), label, text_ft_size,
                                          colors[int(color_idx % 5)])
        color_idx += 1

    return out_image
    pass


def draw_counting_time(image, polylines, counter_output):
    # polylines = [[polyline_id, x, y, x, y, x, y], ...]
    colors = [[18, 153, 255], [31, 102, 156], [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [255, 0, 255]]
    color_idx = 2
    out_image = image
    # ========imagefont===============
    font = ImageFont.truetype("data/simhei.ttf", 22)
    out_image = Image.fromarray(out_image)
    draw = ImageDraw.Draw(out_image)
    # =======ft==================
    # counting_ft = PutChineseText("data/simhei.ttf")
    # text_ft_size = 22

    for polyline in polylines:
        if len(polyline) < 5:
            continue
        pass

        pt1 = (polyline[1], polyline[2])
        pt2 = (polyline[3], polyline[4])
        id = polyline[0]

        # label = "enter: " + str(counter_output.get(id).get("enter")) + " exit: " + str(
        #    counter_output.get(id).get("exit"))
        # label = "进: " + str(counter_output.get(id).get("enter")) + " 出: " + str(
        #     counter_output.get(id).get("exit"))
        # label = "产品个数: " + str(counter_output.get(id).get("enter"))
        label = "操作次数: " + str(counter_output.get(id).get("enter"))
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)
        left = pt1[0] + (pt2[0] - pt1[0]) / 2 - text_size[0][0] / 2
        top = pt1[1] + (pt2[1] - pt1[1]) / 2 - text_size[0][1] / 2

        # ===========english==========================
        # cv2.putText(image, label, (int(left), int(top)), cv2.FONT_HERSHEY_COMPLEX, 0.6, colors[int(color_idx % 5)], 1)

        # ===============ft============================
        # out_image = counting_ft.draw_text(out_image, (int(left), int(top)), label, text_ft_size, colors[int(color_idx % 5)])

        # ====imagefont
        draw.text((int(left), int(top)), label, tuple(colors[int(color_idx % 5)]), font=font)
        color_idx += 1
    out_image = np.asarray(out_image)

    return out_image


label_during = ""

def draw_working_time_rtsp(image, TMCounter, end_work_flag):
    label_end = ""
    global label_during
    # font = ImageFont.truetype("data/simhei.ttf", 42)
    font = ImageFont.truetype("data/simhei.ttf", 42)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    if TMCounter.working_now:
        # ======================english====================================
        # cv2.putText(image, "start time: " + start_work_time, (30, 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, [0,0,255], 2)

        # now_time =  time.strftime("%H:%M:%S",time.localtime())
        # time_diff_str = processing.compute_time(start_work_time, now_time) # time diff with start time
        # label_end = "end time: " + now_time #time_diff_str
        # cv2.putText(image, label_end, (700, 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, [0,0,255], 2)

        # ======================chinese====================================
        label = "开始时间: " + TMCounter.start_work_time
        left, top = 30, 30
        draw.text((int(left), int(top)), label, tuple([0, 0, 255]), font=font)
        # #system time
        now_time = time.strftime("%H:%M:%S", time.localtime())
        time_diff_str = processing.compute_time(TMCounter.start_work_time, now_time)  # time diff with start time

        label_end = "正在工作: " + time_diff_str
        left, top = 700, 30
        draw.text((int(left), int(top)), label_end, tuple([0, 0, 255]), font=font)

    if end_work_flag:
        # ===============english=================================
        # working_now = False
        # record_working_time = True
        # total_time = processing.compute_time(start_work_time, end_work_time)
        # label_during = "work time: " + total_time
        # cv2.putText(image, label_end, (1400, 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, [0,0,255], 2)
        # ====================chinese============================
        TMCounter.show_last_working_time = True
        # total_time = processing.compute_time(start_work_time, end_work_time)
        total_time_str, total_time_sec = TMCounter.get_start_end_work_time_rtsp()
        label_during = "工时统计: " + total_time_str
        left, top = 1400, 30
        draw.text((int(left), int(top)), label_during, tuple([0, 0, 255]), font=font)

    if TMCounter.show_last_working_time:
        # =============english=====================
        # cv2.putText(image, label_during, (1400, 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, [0,0,255], 2)
        left, top = 1400, 30
        draw.text((int(left), int(top)), label_during, tuple([0, 0, 255]), font=font)

    image = np.asarray(image)

    return image


label_during = ""


def draw_working_time(image, TMCounter, end_work_flag, frame_idx):
    label_end = ""
    global label_during
    # font = ImageFont.truetype("data/simhei.ttf", 42)
    font = ImageFont.truetype("data/simhei.ttf", 42)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    if TMCounter.working_now:
        # ======================english====================================
        # cv2.putText(image, "start time: " + start_work_time, (30, 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, [0,0,255], 2)

        # now_time =  time.strftime("%H:%M:%S",time.localtime())
        # time_diff_str = processing.compute_time(start_work_time, now_time) # time diff with start time
        # label_end = "end time: " + now_time #time_diff_str
        # cv2.putText(image, label_end, (700, 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, [0,0,255], 2)

        # ======================chinese====================================
        label = "开始时间: " + TMCounter.start_work_time
        left, top = 30, 30
        draw.text((int(left), int(top)), label, tuple([0, 0, 255]), font=font)
        # #system time
        # now_time =  time.strftime("%H:%M:%S",time.localtime())
        # time_diff_str = processing.compute_time(start_work_time, now_time) # time diff with start time

        # #frame diff time
        time_diff = (frame_idx - TMCounter.start_work_frame_idx) / float(TMCounter.fps)
        time_diff_str = processing.sec2hms(time_diff)

        label_end = "正在工作: " + time_diff_str
        left, top = 700, 30
        draw.text((int(left), int(top)), label_end, tuple([0, 0, 255]), font=font)

    if end_work_flag:
        # ===============english=================================
        # working_now = False
        # record_working_time = True
        # total_time = processing.compute_time(start_work_time, end_work_time)
        # label_during = "work time: " + total_time
        # cv2.putText(image, label_end, (1400, 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, [0,0,255], 2)
        # ====================chinese============================
        TMCounter.show_last_working_time = True
        # total_time = processing.compute_time(start_work_time, end_work_time)
        total_time_str, total_time_sec = TMCounter.get_start_end_work_time()
        label_during = "工时统计: " + total_time_str
        left, top = 1400, 30
        draw.text((int(left), int(top)), label_during, tuple([0, 0, 255]), font=font)

    if TMCounter.show_last_working_time:
        # =============english=====================
        # cv2.putText(image, label_during, (1400, 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, [0,0,255], 2)
        left, top = 1400, 30
        draw.text((int(left), int(top)), label_during, tuple([0, 0, 255]), font=font)

    image = np.asarray(image)

    return image


def draw_time_panel_rtsp(image_cpu, TMCounter, polylines, PLCounter, end_work_flag, min_time, max_time):
    # add background image
    xmin, ymin = 1400, 100  # left top
    x_width, y_height = 410, 500
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # red, blue, green
    polygon_list = [[xmin, ymin], [xmin + x_width, ymin], [xmin + x_width, ymin + y_height], [xmin, ymin + y_height]]
    image_copy = image_cpu.copy()
    cv2.fillConvexPoly(image_copy, np.array(polygon_list), [255, 255, 255])  # 18, 153, 255
    cv2.addWeighted(image_cpu, 0.1, image_copy, 1 - 0.1, 0.0, image_cpu)

    # head
    # occupancy_ft = PutChineseText("data/simhei.ttf") #data/kaiti.ttf
    occupancy_ft = PutChineseText("data/simhei.ttf")  # data/kaiti.ttf
    text_ft_size = 21
    label = "\t操作次数\t|\t开始时间\t|\t工作时长"
    image_cpu = occupancy_ft.draw_text(image_cpu, (xmin + 1, ymin + 10), label, text_ft_size, colors[1])  # 0, 139, 0

    # dict to save the time information
    line_id = polylines[0][0]  # first line
    # print("line_id " + str(line_id))
    margin = 50  # head to first line
    limit_item_number = 10
    if end_work_flag:
        op_num = str(PLCounter.get_counter_number().get(line_id).get("enter"))
        start_time = TMCounter.start_work_time
        work_time_str, work_time_sec = TMCounter.get_start_end_work_time_rtsp()
        TMCounter.time_dict.append({'op_num': op_num, 'start_time': start_time, 'work_time_str': work_time_str,
                                    'work_time_sec': work_time_sec})

    if len(TMCounter.time_dict) > limit_item_number:
        del TMCounter.time_dict[:limit_item_number]

    for item in TMCounter.time_dict:
        line = "\t\t\t" + item['op_num'] + "\t\t\t\t" + item['start_time'] + "\t\t" + item['work_time_str']

        if item['work_time_sec'] > max_time:
            image_cpu = occupancy_ft.draw_text(image_cpu, (xmin + 1, ymin + margin), line, text_ft_size, colors[0])
        elif item['work_time_sec'] < min_time:
            image_cpu = occupancy_ft.draw_text(image_cpu, (xmin + 1, ymin + margin), line, text_ft_size, colors[0])
        else:
            image_cpu = occupancy_ft.draw_text(image_cpu, (xmin + 1, ymin + margin), line, text_ft_size, colors[2])

        margin += 40

    return image_cpu


def draw_time_panel(image_cpu, TMCounter, polylines, PLCounter, end_work_flag):
    # add background image
    xmin, ymin = 1400, 100  # left top
    x_width, y_height = 410, 500
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # red, blue, green
    polygon_list = [[xmin, ymin], [xmin + x_width, ymin], [xmin + x_width, ymin + y_height], [xmin, ymin + y_height]]
    image_copy = image_cpu.copy()
    cv2.fillConvexPoly(image_copy, np.array(polygon_list), [255, 255, 255])  # 18, 153, 255
    cv2.addWeighted(image_cpu, 0.1, image_copy, 1 - 0.1, 0.0, image_cpu)

    # head
    # occupancy_ft = PutChineseText("data/simhei.ttf") #data/kaiti.ttf
    occupancy_ft = PutChineseText("data/simhei.ttf")  # data/kaiti.ttf
    text_ft_size = 21
    label = "\t操作次数\t|\t开始时间\t|\t工作时长"
    image_cpu = occupancy_ft.draw_text(image_cpu, (xmin + 1, ymin + 10), label, text_ft_size, [0, 0, 255])  # 0, 139, 0

    # dict to save the time information
    line_id = polylines[0][0]  # first line
    margin = 50  # head to first line
    limit_item_number = 10
    if end_work_flag:
        op_num = str(PLCounter.get_counter_number().get(line_id).get("enter"))
        start_time = TMCounter.start_work_time
        work_time_str, work_time_sec = TMCounter.get_start_end_work_time()
        TMCounter.time_dict.append({'op_num': op_num, 'start_time': start_time, 'work_time_str': work_time_str,
                                    'work_time_sec': work_time_sec})

    if len(TMCounter.time_dict) > limit_item_number:
        del TMCounter.time_dict[:limit_item_number]

    for item in TMCounter.time_dict:
        line = "\t\t\t" + item['op_num'] + "\t\t\t\t" + item['start_time'] + "\t\t" + item['work_time_str']
        if item['work_time_sec'] > 45:
            image_cpu = occupancy_ft.draw_text(image_cpu, (xmin + 1, ymin + margin), line, text_ft_size, colors[0])
        elif item['work_time_sec'] < 25:
            image_cpu = occupancy_ft.draw_text(image_cpu, (xmin + 1, ymin + margin), line, text_ft_size, colors[0])
        else:
            image_cpu = occupancy_ft.draw_text(image_cpu, (xmin + 1, ymin + margin), line, text_ft_size, colors[2])
        margin += 40

    return image_cpu


label_during = ""


def draw_top_pose_time(image, pose_time, end_work):
    label_end = ""
    global label_during
    # # font = ImageFont.truetype("data/simhei.ttf", 42)
    # font = ImageFont.truetype("data/simhei.ttf", 42)
    # image = Image.fromarray(image)
    # draw = ImageDraw.Draw(image)
    occupancy_ft = PutChineseText("data/simhei.ttf")  # data/kaiti.ttf
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # red, blue, green
    text_ft_size = 42
    # image = image.copy()
    if pose_time.start_work:
        label = "开始时间: " + pose_time.start_work_time
        left, top = 30, 30
        # draw.text((int(left), int(top)), label, tuple([0, 0, 255]), font=font)
        image = occupancy_ft.draw_text(image, (int(left), int(top)), label, text_ft_size, colors[0])
        # #system time
        now_time = time.strftime("%H:%M:%S", time.localtime())
        time_diff_str = processing.compute_time(pose_time.start_work_time, now_time)  # time diff with start time

        label_end = "正在工作: " + time_diff_str
        left, top = 700, 30
        # draw.text((int(left), int(top)), label_end, tuple([0, 0, 255]), font=font)
        image = occupancy_ft.draw_text(image, (int(left), int(top)), label_end, text_ft_size, colors[0])

    if end_work:
        # ===============english=================================
        # working_now = False
        # record_working_time = True
        # total_time = processing.compute_time(start_work_time, end_work_time)
        # label_during = "work time: " + total_time
        # cv2.putText(image, label_end, (1400, 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, [0,0,255], 2)
        # ====================chinese============================
        pose_time.show_last_working_time = True
        # total_time = processing.compute_time(start_work_time, end_work_time)
        total_time_str, total_time_sec = pose_time.get_start_end_work_time_rtsp()
        label_during = "工时统计: " + total_time_str
        left, top = 1400, 30
        # draw.text((int(left), int(top)), label_during, tuple([0, 0, 255]), font=font)
        image = occupancy_ft.draw_text(image, (int(left), int(top)), label_during, text_ft_size, colors[0])

    if pose_time.show_last_working_time:
        # =============english=====================
        # cv2.putText(image, label_during, (1400, 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, [0,0,255], 2)
        left, top = 1400, 30
        # draw.text((int(left), int(top)), label_during, tuple([0, 0, 255]), font=font)
        image = occupancy_ft.draw_text(image, (int(left), int(top)), label_during, text_ft_size, colors[0])
        # print(2222222222222222222)
    image = np.asarray(image)
    # print(image)

    return image


def draw_right_pose_time(image_cpu, pose_time, pose_counter, end_work, min_time, max_time):
    # add background image
    xmin, ymin = 1400, 100  # left top
    x_width, y_height = 410, 500
    # xmin, ymin = 100, 100  # left top
    # x_width, y_height = 410, 500
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # red, blue, green
    polygon_list = [[xmin, ymin], [xmin + x_width, ymin], [xmin + x_width, ymin + y_height], [xmin, ymin + y_height]]
    image_copy = image_cpu.copy()
    cv2.fillConvexPoly(image_copy, np.array(polygon_list), [255, 255, 255])  # 18, 153, 255
    cv2.addWeighted(image_cpu, 0.1, image_copy, 1 - 0.1, 0.0, image_cpu)

    # head
    # occupancy_ft = PutChineseText("data/simhei.ttf") #data/kaiti.ttf
    occupancy_ft = PutChineseText("data/simhei.ttf")  # data/kaiti.ttf
    text_ft_size = 21
    label = "\t操作次数\t|\t开始时间\t|\t工作时长"
    image_cpu = occupancy_ft.draw_text(image_cpu, (xmin + 1, ymin + 10), label, text_ft_size, colors[1])  # 0, 139, 0

    # dict to save the time information
    line_id = 1  # first line
    margin = 50  # head to first line
    limit_item_number = 10
    if end_work:
        # op_num = str(PLCounter.get_counter_number().get(line_id).get("enter"))
        op_num = str(pose_time.op_num)
        start_time = pose_time.start_work_time
        work_time_str, work_time_sec = pose_time.get_start_end_work_time_rtsp()
        pose_time.time_dict.append({'op_num': op_num, 'start_time': start_time, 'work_time_str': work_time_str,
                                    'work_time_sec': work_time_sec})

    if len(pose_time.time_dict) > limit_item_number:
        del pose_time.time_dict[:limit_item_number]

    for item in pose_time.time_dict:
        line = "\t\t\t" + item['op_num'] + "\t\t\t\t" + item['start_time'] + "\t\t" + item['work_time_str']

        if item['work_time_sec'] > max_time:
            image_cpu = occupancy_ft.draw_text(image_cpu, (xmin + 1, ymin + margin), line, text_ft_size, colors[0])
        elif item['work_time_sec'] < min_time:
            image_cpu = occupancy_ft.draw_text(image_cpu, (xmin + 1, ymin + margin), line, text_ft_size, colors[0])
        else:
            image_cpu = occupancy_ft.draw_text(image_cpu, (xmin + 1, ymin + margin), line, text_ft_size, colors[2])

        margin += 40

    return image_cpu


def draw_polygons(image, polygons):
    # polygons = [[polygon_id, x, y, x, y, x, y], ...]
    # colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [255, 0, 255]]
    colors = [[0, 255, 0], [0, 255, 255], [255, 0, 255], [255, 0, 0], [255, 255, 0], [255, 0, 255], [0, 0, 255]]
    color_idx = 0

    for polygon in polygons:
        len_polygon = len(polygon)

        polygon_list = []
        for i in range(1, len_polygon, 2):
            polygon_list.append([int(polygon[i]), int(polygon[i + 1])])
        pass

        #  # 加上下面这段两边颜色一样
        # id = int(polygon[0])
        #
        # if id == 10 or id == 12:
        #     color_idx = 0
        # if id == 11:
        #     color_idx = 1

        image_copy = image.copy()

        # fillConvexPoly: 可以用来填充凸多边形， 只需要提供凸多边形的顶点即可
        # fillPoly: 可以用来填充任意形状的图型.可以用来绘制多边形,工作中也经常使用非常多个边来近似的画一条曲线.fillPoly可以一次填充多个图型.
        cv2.fillConvexPoly(image_copy, np.array(polygon_list), colors[int(color_idx % 5)])
        cv2.addWeighted(image, 0.5, image_copy, 1 - 0.5, 0.0, image)
        color_idx += 1
    pass


def draw_occupancy_counting(image, polygons, counter_output, alarm_person_num):
    # polygons = [[polyline_id, x, y, x, y, x, y], ...]
    colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [255, 0, 255]]
    color_idx = 0
    out_image = image
    polygons_ids = []
    # get the polygons_ids for alarm
    for polygon in polygons:
        if len(polygon) < 7:
            continue
        pass
        polygons_ids.append(int(polygon[0]))
    pass

    for polygon in polygons:
        if len(polygon) < 7:
            continue
        pass

        alarm_flag = 0
        pt1 = (int(polygon[1]), int(polygon[2]))
        pt2 = (int(polygon[3]), int(polygon[4]))
        id = int(polygon[0])
        text_ft_size = 21

        if id == polygons_ids[0]:
            occupancy_ft = PutChineseText("data/simhei.ttf")
            label = "密度统计: " + str(counter_output.get(id))
            if counter_output.get(id) >= alarm_person_num:
                label_alarm = "报警: " + str(counter_output.get(id))
                alarm_flag = 1

        elif id == polygons_ids[1]:
            occupancy_ft = PutChineseText("data/simhei.ttf")
            label = "闯入检测: " + str(counter_output.get(id))
            if counter_output.get(id) > 0:
                label_alarm = "报警: " + str(counter_output.get(id))
                alarm_flag = 1
        else:
            label = "闯入检测: " + str(counter_output.get(id))
        pass

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)
        left = pt1[0] + (pt2[0] - pt1[0]) / 2 - text_size[0][0] / 2
        top = pt1[1] + (pt2[1] - pt1[1]) / 2 - text_size[0][1] / 2
        # cv2.putText(image, label, (int(left), int(top)), cv2.FONT_HERSHEY_COMPLEX, 0.6, colors[2], 1)
        out_image = occupancy_ft.draw_text(out_image, (int(left), int(top)), label, text_ft_size, colors[2])

        if alarm_flag == 1:
            text_ft_size = 24  # larger font
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)
            left = pt1[0] + (pt2[0] - pt1[0]) / 2 - text_size[0][0] / 2
            top = pt1[1] + (pt2[1] - pt1[1]) / 2 + 20
            # cv2.putText(image, label_alarm, (int(left), int(top)), cv2.FONT_HERSHEY_COMPLEX, 0.7, colors[0], 1)
            out_image = occupancy_ft.draw_text(out_image, (int(left), int(top)), label_alarm, text_ft_size, colors[0])
            alarm_flag = 0
        pass

        color_idx += 1
    return out_image
    pass


def draw_occupancy_time(image, polygons, counter_output):
    # polygons = [[polyline_id, x, y, x, y, x, y], ...]
    colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [255, 0, 255]]
    color_idx = 0
    # =============ft=================
    # counting_ft = PutChineseText("data/simhei.ttf")
    counting_ft = PutChineseText("data/simhei.ttf")
    text_ft_size = 22

    for polygon in polygons:
        if len(polygon) < 7:
            continue
        pass
        ###=================english===========================
        # pt1 = (int(polygon[1]), int(polygon[2]))
        # pt2 = (int(polygon[3]), int(polygon[4]))
        # id = int(polygon[0])
        # label = "occupancy: " + str(counter_output.get(id))
        # text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)
        # left = pt1[0] + (pt2[0] - pt1[0]) / 2 - text_size[0][0] / 2
        # top = pt1[1] + (pt2[1] - pt1[1]) / 2 - text_size[0][1] / 2
        # cv2.putText(image, label, (int(left), int(top)), cv2.FONT_HERSHEY_COMPLEX, 0.6, colors[int(color_idx % 5)], 1)
        # color_idx += 1

        ###=================chinese===========================
        pt1 = (int(polygon[1]), int(polygon[2]))
        pt2 = (int(polygon[3]), int(polygon[4]))
        left = pt1[0] + (pt2[0] - pt1[0]) / 2
        top = pt1[1] + (pt2[1] - pt1[1]) / 2
        id = int(polygon[0])
        label = ""
        color = tuple(colors[int(color_idx % 5)])

        if id == 10 or id == 12:
            label = "人数: " + str(counter_output.get(id))
            color_idx = 0
            color = tuple(colors[int(color_idx % 5)])
        if id == 11:
            color_idx = 0
            label = "电机数: " + str(counter_output.get(id))
            color = tuple(colors[int(color_idx % 5)])

        # ================imagefont=======================
        # font = ImageFont.truetype("data/simhei.ttf", 20)
        # image = Image.fromarray(image)
        # draw = ImageDraw.Draw(image)
        # draw.text((int(left), int(top)), label,  color, font=font)
        # image = np.asarray(image)

        # =======================ft======================
        image = counting_ft.draw_text(image, (int(left), int(top)), label, text_ft_size, colors[int(color_idx % 5)])

    pass
    return image


def draw_occupancy_alarm(image, polygons, counter_output):
    # polygons = [[polyline_id, x, y, x, y, x, y], ...]
    colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [255, 0, 255]]
    color_idx = 0

    for polygon in polygons:
        if len(polygon) < 7:
            continue
        pass

        pt1 = (int(polygon[1]), int(polygon[2]))
        pt2 = (int(polygon[3]), int(polygon[4]))
        id = int(polygon[0])
        label = "occupancy: " + str(counter_output.get(id))

        if counter_output.get(id) > 0:
            label = "occupancy: " + str(counter_output.get(id)) + " ALARM"

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1.1, 1)
        left = pt1[0] + (pt2[0] - pt1[0]) / 2 - text_size[0][0] / 2
        top = pt1[1] + (pt2[1] - pt1[1]) / 2 - text_size[0][1] / 2
        cv2.putText(image, label, (int(left), int(top)), cv2.FONT_HERSHEY_COMPLEX, 0.6, colors[int(color_idx % 5)], 1)
        color_idx += 1
    pass


def merge_imgs(img1, img2, xmin, xmax, ymin, ymax):
    dst_channels = cv2.split(img1)
    src_channels = cv2.split(img2)
    b, g, r, a = cv2.split(img2)

    for i in range(3):
        dst_channels[i][ymin:ymax, xmin:xmax] = dst_channels[i][ymin:ymax, xmin:xmax] * (255.0 - a) / 255
        dst_channels[i][ymin:ymax, xmin:xmax] += np.array(src_channels[i] * (a / 255.), dtype=np.uint8)
    pass

    out = cv2.merge(dst_channels)
    return out


def add_logo(image):
    logo_path = "data/logo.png"
    if not os.path.exists(logo_path):
        print("logo image not existed")
        sys.exit()

    logo_img = cv2.imread(logo_path, -1)
    ss = logo_img.shape
    re_width = 400
    re_height = int(float(ss[0]) / ss[1] * re_width)
    logo_img = cv2.resize(logo_img, (re_width, re_height))
    out_image = merge_imgs(image, logo_img, 10, 10 + re_width, image.shape[0] - 10 - re_height, image.shape[0] - 10)

    return out_image
