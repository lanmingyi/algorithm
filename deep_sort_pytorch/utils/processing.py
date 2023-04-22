#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import datetime


def detection_filter(detections, names_list):
    dets = []

    for detection in detections[0]:
        category = detection['name']
        # xmin = int(detection['left'])
        # ymin = int(detection['top'])
        # xmax = int(detection['right'])
        # ymax = int(detection['bottom'])

        if category in names_list:
            dets.append(detection)
        pass
    pass

    return dets


def seperate_detection_filter(detections, names_list):
    dianji_dets = []  # names_list[0]
    jichupan_dets = []  # names_list[1]

    for detection in detections[0]:
        category = detection['name']

        if category in names_list[0]:
            dianji_dets.append(detection)
        elif category in names_list[1]:
            jichupan_dets.append(detection)
        else:
            pass

        pass
    pass

    return dianji_dets, jichupan_dets


def is_in_polygon(center, poly):
    poly_x = poly[::2]
    poly_y = poly[1::2]
    res = False
    i = -1
    l = len(poly_x)
    j = l - 1
    while i < l - 1:
        i += 1
        # print(i, [poly_x[i], poly_y[i]], j, [poly_x[j], poly_y[j]])
        if ((poly_x[i] <= center[0] and center[0] < poly_x[j]) or (poly_x[j] <= center[0] and center[0] < poly_x[i])):
            if (center[1] < (poly_y[j] - poly_y[i]) * (center[0] - poly_x[i]) / (poly_x[j] - poly_x[i]) + poly_y[i]):
                res = not res
        j = i
    return res


def compute_time(start_time, end_time):
    start_time =  datetime.datetime.strptime(start_time, "%H:%M:%S")
    end_time = datetime.datetime.strptime(end_time,"%H:%M:%S")
    time_diff_str = str(end_time - start_time)
    return time_diff_str


def sec2hms(seconds):
    seconds_int = int(seconds)
    time_str =str(datetime.timedelta(seconds = seconds_int))
    return time_str


def polygons_counter(tracker_result, polygons):
    polygons_count = {}
    for polygon in polygons:
        poly_id = int(polygon[0])
        polygons_count.update({poly_id: 0})
    pass

    # state = 2: confirmed
    # update_times: Without detection information, the number of tracking positions is predicted.
    # time_since_update : Total number of frames since last measurement update.
    for track_id, track_info in tracker_result.items():
        for frame_id, track_data in track_info.items():
            if (int(track_data['state'])) != 2 or (int(track_data['update_times'])) > 0:
                continue
            pass

            track_box = track_data['track_box']
            if len(track_box) != 0:
                xmin = int(track_box[0])
                ymin = int(track_box[1])
                xmax = int(track_box[2])
                ymax = int(track_box[3])

                center_x = (xmin + xmax) / 2.0
                center_y = (ymin + ymax) / 2.0

                center = (center_x, center_y)

                for polygon in polygons:
                    poly_id = int(polygon[0])
                    poly = polygon[1:]

                    if is_in_polygon(center, poly):
                        polygons_count[poly_id] += 1
                    pass
                pass
            pass
        pass
    pass

    return polygons_count