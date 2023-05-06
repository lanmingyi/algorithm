"""
Copyright (c) 2021 Bright Mind. All rights reserved.
Written by MingYi Lan
"""

import time
import datetime
# from utils import processing
from . import processing


class PoseTime:
    def __init__(self, touch_zones, fps, trust_frames=10):
        self.start_work_time = None  # record starting working time
        self.end_work_time = None
        self.start_touch_time = None
        self.end_touch_time = None
        self.diff_time_secs = 0
        self.op_num = 0
        self.start_work_frame_idx = 0
        self.end_work_frame_idx = None
        self.show_last_working_time = False  # show the last working time
        self.start_work = False  # is working
        self.end_work = False
        self.save_start_img = False
        self.save_end_img = False
        self.polygons_frames = {}  # occur number of frames in polygons
        self.trust_frames = trust_frames
        self.fps = fps
        self.time_dict = []
        self.touchshowdic = {}
        self.onlytouch = {}
        self.touch_zones = touch_zones
        # self.touch_zones = [
        #     [600, 350, 900, 350, 900, 600, 600, 600],
        #     # [1500, 400, 1700, 400, 1700, 600, 1500, 600],
        # ]

    def occur_in_zones(self, person_dets, dianji_dets, polygons):
        polygons_count = {}  # 数量

        # 是否连续在一个polygon中出现
        in_polygon_one = False
        in_polygon_two = False
        in_polygon_three = False

        index_one = 1  # polygon 1
        poly_id_one = int(polygons[index_one - 1][0])

        index_two = 2
        poly_id_two = int(polygons[index_two - 1][0])

        index_three = 3  # polygon 3
        poly_id_three = int(polygons[index_three - 1][0])

        for polygon in polygons:
            poly_id = int(polygon[0])
            polygons_count.update({poly_id: 0})
        pass

        for det in person_dets:
            xmin = int(det['left'])
            ymin = int(det['top'])
            xmax = int(det['right'])
            ymax = int(det['bottom'])

            center_x = (xmin + xmax) / 2.0
            center_y = (ymin + ymax) / 2.0
            center = (center_x, center_y)
            # polygon 1
            poly = polygons[index_one - 1][1:]
            if processing.is_in_polygon(center, poly):
                polygons_count[poly_id_one] += 1
                in_polygon_one = True
            pass

            poly = polygons[index_three - 1][1:]
            if processing.is_in_polygon(center, poly):
                polygons_count[poly_id_three] += 1
                in_polygon_three = True
            pass
        pass

        for det in dianji_dets:
            xmin = int(det['left'])
            ymin = int(det["top"])
            xmax = int(det['right'])
            ymax = int(det['bottom'])

            center_x = (xmin + xmax) / 2.0
            center_y = (ymin + ymax) / 2.0
            center = (center_x, center_y)
            # polygon 2
            poly = polygons[index_two - 1][1:]
            if processing.is_in_polygon(center, poly):
                polygons_count[poly_id_two] += 1
                in_polygon_two = True
            pass
        pass

        # 统计出现次数
        if in_polygon_one:
            self.polygons_frames.setdefault(poly_id_one, 0)
            self.polygons_frames[poly_id_one] += 1
        else:
            self.polygons_frames[poly_id_one] = 0

        if in_polygon_two:
            self.polygons_frames.setdefault(poly_id_two, 0)
            self.polygons_frames[poly_id_two] += 1
        else:
            self.polygons_frames[poly_id_two] = 0

        if in_polygon_three:
            self.polygons_frames.setdefault(poly_id_three, 0)
            self.polygons_frames[poly_id_three] += 1
        else:
            self.polygons_frames[poly_id_three] = 0

        # working 时重置
        if self.start_work is True:
            self.polygons_frames = {}

        return polygons_count

    def judge_start_time(self, polygons_count, polygons, frame_idx):
        flag_dict = {}
        for polygon in polygons:
            pgon_id = int(polygon[0])
            num = polygons_count.get(pgon_id)  # 每一帧中出现物体的个数
            frame_num = self.polygons_frames.get(pgon_id, 0)  # 连续一段时间出现的次数
            if num >= 1 and frame_num >= self.trust_frames:
                flag_dict.update({pgon_id: True})
            else:
                flag_dict.update({pgon_id: False})
            pass
        pass

        start_time_flag = True  # True 所有的polygons要有人，False其中一个polygons有人即可
        # strftime:f表示format，表示格式化，给定一个时间对象和输出格式，返回一个时间字符串
        start_time = time.strftime("%H:%M:%S", time.localtime())

        for key, value in flag_dict.items():
            start_time_flag = start_time_flag and value  # and , or
        pass

        if start_time_flag and not self.start_work:
            self.start_work = True
            self.start_work_time = start_time
            self.start_work_frame_idx = frame_idx
            self.save_start_img = True

        return self.start_work

    def get_start_end_work_time(self):
        # fps = frame / time      time = frame / fps
        total_time = (self.end_work_frame_idx - self.start_work_frame_idx) / float(self.fps)
        total_time_sec = int(total_time)
        total_time_str = processing.sec2hms(total_time)
        return total_time_str, total_time_sec

    def get_start_end_work_time_rtsp(self):
        # strptime:p表示parse，表示分析的意思，给定一个时间字符串和分析模式，返回一个时间对象
        start_time = datetime.datetime.strptime(self.start_work_time, "%H:%M:%S")
        end_time = datetime.datetime.strptime(self.end_work_time, "%H:%M:%S")
        total_time_str = str(end_time - start_time)
        total_time_secs = (end_time - start_time).seconds
        return total_time_str, total_time_secs

    def get_start_time(self):
        return self.start_work_time

    def get_now_time(self, frame_idx):
        diff_time = (frame_idx - self.start_work_frame_idx) / float(self.fps)
        diff_time_str = processing.sec2hms(diff_time)
        return diff_time_str

    def get_start2now_time(self, frame_idx):
        diff_time = (frame_idx - self.start_work_frame_idx) / float(self.fps)
        return int(diff_time)

    def get_start2now_time_rtsp(self):
        start_time = datetime.datetime.strptime(self.start_work_time, "%H:%M:%S")
        now_time = datetime.datetime.strptime(time.strftime("%H:%M:%S", time.localtime()), "%H:%M:%S")
        diff_time = (now_time - start_time).seconds
        return diff_time

    # list[start:end:step] 起始位置、结束位置、步长
    """
    射线法就是以判断点开始，向右（或向左）的水平方向作一射线，计算该射线与多边形每条边的交点个数，如果交点个数为奇数，则点位于多边形内，偶数则在多边形外。该算法对于复合多边形也能正确判断。
    射线法的关键是正确计算射线与每条边是否相交。并且规定线段与射线重叠或者射线经过线段下端点属于不相交。首先排除掉不相交的情况
    函数里求交的部分就是利用两个三角形的比例关系求出交点在起点的左边还是右边
    """

    @staticmethod
    def is_in_zone(center, poly):
        poly_x = poly[::2]
        poly_y = poly[1::2]
        res = False
        i = -1
        l = len(poly_x)
        j = l - 1
        while i < l - 1:
            i += 1
            if (poly_x[i] <= center[0] < poly_x[j]) or (poly_x[j] <= center[0] < poly_x[i]):
                # 相似三角形
                if center[1] < (poly_y[j] - poly_y[i]) * (center[0] - poly_x[i]) / (poly_x[j] - poly_x[i]) + poly_y[i]:
                    res = not res
            j = i
        return res

    def is_touch_real(self, keypoints, frame_idx):
        if keypoints is not None:
            for id, keypoints in enumerate(keypoints):

                right_hand_point = keypoints[4, :-1]
                left_hand_point = keypoints[7, :-1]
                zone_flags = [0 for i in range(len(self.touch_zones))]
                # print(zone_flags)
                for i in range(len(self.touch_zones)):
                    # zone_flags里存True和False
                    zone_flags[i] = self.is_in_zone(left_hand_point, self.touch_zones[i]) \
                                    or self.is_in_zone(right_hand_point, self.touch_zones[i])
                # print(111111111111111111111111111111111)
                # print(type(zone_flags))
                # print(zone_flags)

                flags = [0 for i in range(len(self.touch_zones))]
                start_time = time.strftime("%H:%M:%S", time.localtime())
                touch_show_label = ""
                for i in range(len(self.touch_zones)):
                    if zone_flags[i]:
                        flags[i] = 1
                        # touch_show_label = "触摸区域" + str(i + 1)
                        touch_show_label = "开始工作"
                        self.start_work = True
                        self.start_touch_time = time.strftime("%H:%M:%S", time.localtime())
                        self.onlytouch[id] = id
                        # print(self.onlytouch)
                    else:
                        flags[i] = 0
                        self.end_work = True
                        self.end_touch_time = time.strftime("%H:%M:%S", time.localtime())
                if not self.start_work:
                    self.start_work_time = start_time
                    self.start_work_frame_idx = frame_idx

                if self.start_work:
                    # print(111111111111)
                    start_touch_time = datetime.datetime.strptime(self.start_touch_time, "%H:%M:%S")
                    print('start_touch_time', start_touch_time)

                    end_touch_time = datetime.datetime.strptime(self.end_touch_time, "%H:%M:%S")
                    print('end_touch_time', end_touch_time)
                    # print(22222222222222)
                    # print(end_touch_time)
                    if end_touch_time > start_touch_time:
                        # print(333333333333)
                        # diff_time = str(end_time - start_time).seconds
                        self.diff_time_secs = abs(end_touch_time - start_touch_time).seconds

                self.touchshowdic[id] = touch_show_label

        return self.start_work, self.end_work, self.diff_time_secs  # , self.touchshowdic
