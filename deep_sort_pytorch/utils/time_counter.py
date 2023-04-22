#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import datetime
# from src.application_util import processing
import processing


class TimeCounter():
    def __init__(self, fps, trust_frames=10):
        self.start_work_time = None  # record starting working  time
        self.end_work_time = None
        self.start_work_frame_idx = 0
        self.end_work_frame_idx = None
        self.show_last_working_time = False  # show the last working time
        self.working_now = False  # is working
        self.save_start_img = False
        self.save_end_img = False
        self.polygons_frames = {}  # occur number of frames in polygons
        self.trust_frames = trust_frames
        self.fps = fps
        self.time_dict = []

    def occur_in_zones(self, person_dets, dianji_dets,  polygons):
        """
        [[{'index': 0, 'name': 'person', 'confident': 0.9997307658195496,
        'left': 525.1392822265625, 'right': 631.076416015625, 'top': 0.3767511248588562,
        'bottom': 206.82791137695312}]]
        """
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
            polygons_count.update({poly_id: 0})  #
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
            poly = polygons[index_one-1][1:]
            if processing.is_in_polygon(center, poly):
                polygons_count[poly_id_one] += 1
                in_polygon_one = True
            pass

            # polygon 3
            poly = polygons[index_three-1][1:]
            if processing.is_in_polygon(center, poly):
                polygons_count[poly_id_three] += 1
                in_polygon_three = True
            pass
        pass

        for det in dianji_dets:
            xmin = int(det['left'])
            ymin = int(det['top'])
            xmax = int(det['right'])
            ymax = int(det['bottom'])

            center_x = (xmin + xmax) / 2.0
            center_y = (ymin + ymax) / 2.0
            center = (center_x, center_y)
            # polygon 2
            poly = polygons[index_two-1][1:]
            if processing.is_in_polygon(center, poly):
                polygons_count[poly_id_two] += 1
                in_polygon_two = True
            pass
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
        if self.working_now is True: self.polygons_frames = {} 

        return polygons_count

    def judge_start_time(self, polygons_count, polygons, frame_idx):
    
        flag_dict = {}
        for polygon in polygons:
            pgon_id = int(polygon[0])
            num = polygons_count.get(pgon_id)  # 每一帧中出现物体的个数
            frame_num = self.polygons_frames.get(pgon_id, 0)  # 连续一段时间出现的次数
            if num >= 1 and frame_num >= self.trust_frames:
                flag_dict.update({pgon_id:True})
            else:
                flag_dict.update({pgon_id:False})
            pass
        pass
        
        start_time_flag = True    # True 所有的polygons要有人，False其中一个polygons有人即可
        start_time = time.strftime("%H:%M:%S", time.localtime())

        for key, value in flag_dict.items(): 
            start_time_flag = start_time_flag and value # and, or
        pass

        if start_time_flag and not self.working_now:
            self.working_now = True
            self.start_work_time = start_time
            self.start_work_frame_idx = frame_idx
            self.save_start_img = True
        
        return self.working_now
    
    def get_start_end_work_time(self):
        total_time = (self.end_work_frame_idx - self.start_work_frame_idx) / float(self.fps)
        total_time_sec = int(total_time)
        total_time_str = processing.sec2hms(total_time)
        return total_time_str, total_time_sec

    def get_start_end_work_time_rtsp(self):
        start_time = datetime.datetime.strptime(self.start_work_time, "%H:%M:%S")
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


if __name__ == '__main__':
    fps = 30
    TRUST_FRAMES = 2
    TMCounter = TimeCounter(fps, trust_frames=TRUST_FRAMES)
    work_time_str, work_time_sec = TMCounter.get_start_end_work_time_rtsp()
    # print(work_time_str)
    # print(work_time_sec)

