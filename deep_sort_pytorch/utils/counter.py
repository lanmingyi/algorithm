import cv2
import pdb
import pickle
import numpy as np

left_vehicle_counter = 0
right_vehicle_counter = 0
bottom_vehicle_counter = 0
counter = 0
deviation = 4.5

LEFT_INTERSECTION_ROI_POSITION = 400
LEFT_INTERSECTION_ROI_START = 300
LEFT_INTERSECTION_ROI_END = 550

# RIGHT_INTERSECTION_ROI_POSITION = 1000
# RIGHT_INTERSECTION_ROI_START = 0
# RIGHT_INTERSECTION_ROI_END = 280

RIGHT_INTERSECTION_ROI_POSITION = 490
RIGHT_INTERSECTION_ROI_START = 100
RIGHT_INTERSECTION_ROI_END = 400

BOTTOM_INTERSECTION_ROI_POSITION = 500
BOTTOM_INTERSECTION_ROI_START = 500
BOTTOM_INTERSECTION_ROI_END = 900

bbox_color = {'ignored-regions': (0xFF, 0x66, 0x00),
              'pedestrian': (0xCC, 0x66, 0x00),
              'people': (0x99, 0x66, 0x00),
              'bicycle': (0x66, 0x66, 0x00),
              'car': (0x33, 0xFF, 0x00),
              'van': (0x00, 0x66, 0x00),
              'truck': (0xFF, 0xFF, 0x00),
              'tricycle': (0xCC, 0xFF, 0x00),
              'awning-tricycle': (0x99, 0xFF, 0x00),
              'bus': (0x66, 0xFF, 0x00),
              'motor': (0x33, 0x66, 0x00),
              'others': (0x00, 0xFF, 0x00)}


def trackers_to_perframe(trackers):
    pass


def counting(image, bbox, show_box=True, show_label=True):
    global left_vehicle_counter
    global right_vehicle_counter
    global bottom_vehicle_counter
    global counter

    for i, bbox in enumerate(bbox):

        x1, y1, x2, y2 = [int(i) for i in bbox]

        # print(bbox)
        bbox_last = bbox  # 应该是bbox的下一帧
        center_now = (((bbox[0]) + (bbox[2])) / 2, ((bbox[1]) + (bbox[3])) / 2)  # 有点强
        center_last = (((bbox_last[0]) + (bbox_last[2])) / 2, ((bbox_last[1]) + (bbox_last[3])) / 2)
        # center_now = (x1 + x2 / 2, y1 + y2 / 2)
        # center_last = (x1 + x2 / 2, y1 + y2 / 2)
        is_online = counting_right(bbox[0], bbox[2], bbox[1], bbox[3], LEFT_INTERSECTION_ROI_POSITION, deviation)

        left_vehicle_is_online = counting_horizontal(center_now, center_last, \
                                                     LEFT_INTERSECTION_ROI_POSITION, \
                                                     LEFT_INTERSECTION_ROI_START, \
                                                     LEFT_INTERSECTION_ROI_END)

        right_vehicle_is_online = counting_horizontal(center_now, center_last, \
                                                      RIGHT_INTERSECTION_ROI_POSITION, \
                                                      RIGHT_INTERSECTION_ROI_START, \
                                                      RIGHT_INTERSECTION_ROI_END)

        bottom_vehicle_is_online = counting_vertical(center_now, center_last, \
                                                     BOTTOM_INTERSECTION_ROI_POSITION, \
                                                     BOTTOM_INTERSECTION_ROI_START, \
                                                     BOTTOM_INTERSECTION_ROI_END)

        if is_online:
            counter += 1
        # if left_vehicle_is_online:
        #     left_vehicle_counter += 1
        if right_vehicle_is_online:
            right_vehicle_counter += 1
        if bottom_vehicle_is_online:
            bottom_vehicle_counter += 1

    # cv2.line(image, \
    #          (BOTTOM_INTERSECTION_ROI_START, BOTTOM_INTERSECTION_ROI_POSITION), \
    #          (BOTTOM_INTERSECTION_ROI_END, BOTTOM_INTERSECTION_ROI_POSITION), \
    #          (0, 0, 0xFF), 5)
    # cv2.line(image, \
    #          (RIGHT_INTERSECTION_ROI_POSITION, RIGHT_INTERSECTION_ROI_START), \
    #          (RIGHT_INTERSECTION_ROI_POSITION, RIGHT_INTERSECTION_ROI_END), \
    #          (0, 0, 0xFF), 5)
    # cv2.line(image, \
    #          (LEFT_INTERSECTION_ROI_POSITION, LEFT_INTERSECTION_ROI_START), \
    #          (LEFT_INTERSECTION_ROI_POSITION, LEFT_INTERSECTION_ROI_END), \
    #          (0, 0, 0xFF), 5)

    counter_info = str(counter)
    left_info = str(left_vehicle_counter)
    right_info = str(right_vehicle_counter)
    bottom_info = str(bottom_vehicle_counter)
    # cv2.putText(image, text=left_info,
    #             org=(LEFT_INTERSECTION_ROI_POSITION + 10, LEFT_INTERSECTION_ROI_START + 20),
    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #             fontScale=1, color=(0, 0, 255), thickness=2)
    cv2.putText(image, text=counter_info,
                org=(RIGHT_INTERSECTION_ROI_POSITION - 60, RIGHT_INTERSECTION_ROI_END - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 255), thickness=2)
    # cv2.putText(image, text=bottom_info,
    #             org=(BOTTOM_INTERSECTION_ROI_START + 10, BOTTOM_INTERSECTION_ROI_POSITION - 20),
    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #             fontScale=1, color=(0, 0, 255), thickness=2)

    return image, counter


def counting_right(left, right, top, bottom, line_position, deviation):
    print(left)
    print(right)
    print((left + right)/2)
    print(line_position)


    vehicle_is_online = False
    if abs((left + right)/2 - line_position) < deviation:
        vehicle_is_online = True
    return vehicle_is_online


def counting_horizontal(center_now, center_last, line_position, line_start, line_end):
    # print(center_now)
    #     # print(center_last)
    #     # print(line_position)

    # print(center_now[0] < line_position)
    # print(center_now[0] == line_position)

    vehicle_is_online = False
    if center_now[1] >= line_start and center_now[1] <= line_end and center_last[1] >= line_start and center_last[
        1] <= line_end:
        if (center_now[0] < line_position and center_last[0] >= line_position) or (center_now[0] > line_position and center_last[0] <= line_position):
        # if center_now[0] <= line_position and center_last[0] >= line_position:
        #     print(22222222222222)
            vehicle_is_online = True
    # print(vehicle_is_online)
    return vehicle_is_online


def counting_vertical(center_now, center_last, line_position, line_start, line_end):
    vehicle_is_online = False
    if center_now[0] >= line_start and center_now[0] <= line_end and center_last[0] >= line_start and center_last[0] <= line_end:
        if (center_now[1] < line_position and center_last[1] >= line_position) or (center_now[1] > line_position and center_last[1] <= line_position):
            vehicle_is_online = True

    return vehicle_is_online


if __name__ == '__main__':
    # bbox_xyxy = [[326, 227, 392, 351]]
    bbox_xyxy_total = [[[484, 349, 536, 393]], [[466, 349, 517, 393]], [[442, 347, 493, 391]], [[431, 346, 482, 391]]]
    img = cv2.imread('test/.jpg')
    counting(img, bbox_xyxy_total)
    # pickle_path = "../output/test_output/tmp.pk"
    # with open(pickle_path, 'rb') as pk_f:
    #     trackers = pickle.load(pk_f)
    #     # print(trackers)
    # # for object_id, object_info in enumerate(trackers):
    # #     print(object_info)
    #
    # path = "../docs/images/1.mp4"
    # vid = cv2.VideoCapture(path)
    # # print(vid)
    # while True:
    #     return_value, frame = vid.read()
    #     # print(len(frame))
    #     if return_value != True:
    #         break
    #     # print(return_value, frame)
    #
    #
    #     result = counting(frame, vid.get(1), trackers)
    #     cv2.imshow('1', frame)
    #     if cv2.waitKey(30) & 0xFF == 'q': break
    #     # cv2.waitKey()


# # Return true if line segments AB and CD intersect
# @staticmethod  # 一般来说，要使用某个类的方法，需要先实例化一个对象再调用方法。而使用@staticmethod或@classmethod，就可以不需要实例化，直接类名.方法名()来调用。
# def intersect(A, B, C, D):  # midpoint（中点）, previous_midpoint（以前的中点）, line[0], line[1]
#     return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
#
# @staticmethod
# def ccw(A, B, C):
#     return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
