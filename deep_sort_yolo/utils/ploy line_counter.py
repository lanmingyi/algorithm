import time


class PolylineCounter:
    def __init__(self, polylines, trust_frames=3, ignore_frames=10):
        # polylines = [[polyline_id, x, y, x, y, x, y], ...]
        self.__cnt_dict = {}
        self.lines = polylines
        self.trust_frames = trust_frames
        self.ignore_frames = ignore_frames
        self.cnt = {}
        self.cross_ids = {}
        self.cross_result = []
        self.ids_do_cnt = []
        self.ids_to_del = []
        self.frame_idx = 0
        self.line_ids = []
        self.end_work_flag = False
        # init values
        for line in polylines:
            line_id = line[0]
            self.line_ids.append(line_id)
            self.cnt.update({line_id: {'enter': 0, 'exit': 0}})
            self.__cnt_dict.update({line_id: {}})
            self.cross_ids.update({line_id: {'enter_ids': [], 'exit_ids': []}})
        pass

    def get_counter_number(self):
        # {
        #     line_id:
        #         {
        #             'enter_num': 100,
        #             'exit_num': 90,
        #         }
        #     ...
        # }
        return self.cnt.copy()
    
    def get_finish_flag(self):

        return self.end_work_flag

    def get_current_crossed_ids(self):
        # {
        #     line_id:
        #         {
        #             'enter_ids':[],
        #             'exit_ids':[],
        #         },
        #     ...
        # }
        return self.cross_ids.copy()

    def free(self):
        self.__cnt_dict.clear()
        self.cnt.clear()
        self.cross_ids.clear()

    def update(self, working_now, tracker_output, frame_idx):
        # print track_output
        self.end_work_flag = False
        self.__update_cnt_dict(tracker_output, frame_idx)
        self.__count(working_now)
        # print self.get_counter_number()
        # print self.get_current_crossed_ids()

    """
    track_output:
    {'det_box': [xmin, ymin, xmax, ymax], 'confident': *, 'state': *, 
    'track_box': [xmin, ymin, xmax, ymax], 'timestamp': '00:00:00', 'det_idx': *, 'update_times': *}
    """
    def __update_cnt_dict(self, track_output, frame_idx):
        self.frame_idx = frame_idx
        self.ids_do_cnt = []
        track_ids = track_output.keys()
        for track_id in track_ids:
            # get(key[,default]):Return the value for key if key is in the dictionary, else default.
            # If default is not given, it defaults to None, so that this method never raises a KeyError.
            current_track_dict = track_output.get(track_id).get(str(frame_idx))
            if current_track_dict is None:
                continue
            if current_track_dict.get('state') != 2 or current_track_dict.get('update_times') > 0:
                if current_track_dict.get('state') == 3:
                    self.ids_to_del.append(track_id)
                continue
            self.ids_do_cnt.append(track_id)
            track_box = current_track_dict.get('track_box')  # [xmin, ymin, xmax, ymax]
            # cx = track_box[0] + track_box[2] / 2.
            # cy = track_box[1] + track_box[3] / 2.
            cx = (track_box[0] + track_box[2]) / 2.
            cy = (track_box[1] + track_box[3]) / 2.
            # updated_times = int(current_track_dict.get('updatetimes'))
            for line_id in self.line_ids:
                line_tracks_dict = self.__cnt_dict.get(line_id)
                if track_id in line_tracks_dict.keys():
                    track_dict_tmp = line_tracks_dict.get(track_id)
                    track_dict_tmp['last_cx'] = track_dict_tmp['current_cx']
                    track_dict_tmp['last_cy'] = track_dict_tmp['current_cy']
                    track_dict_tmp['current_cx'] = cx
                    track_dict_tmp['current_cy'] = cy
                    track_dict_tmp['confirm_frames'] += 1
                else:
                    # created new track_id info
                    track_dict_tmp = \
                        {
                            'last_cx': cx,
                            'last_cy': cy,
                            'current_cx': cx,
                            'current_cy': cy,
                            'confirm_frames': 0,
                            'enter_frame_idxs': [],
                            'exit_frame_idxs': [],
                            'unconfirmed_enter_frame_idxs': [],
                            'unconfirmed_exit_frame_idxs': [],
                        }
                    line_tracks_dict.update({track_id: track_dict_tmp})
                pass
            pass
        pass  # for track_id in track_ids:

    def __count(self, working_now):
        for line, line_id in zip(self.lines, self.line_ids):
            line_tracks_dict = self.__cnt_dict.get(line_id)
            # self.cross_ids.update({line_id: {'enter_ids': [], 'exit_ids': []}})
            line_crossid_dict = self.cross_ids.get(line_id)
            line_crossid_dict['enter_ids'] = []
            line_crossid_dict['exit_ids'] = []

            if line_tracks_dict == {}:
                continue

            # delete useless track_id
            for track_id in self.ids_to_del:
                if track_id in line_tracks_dict:
                    line_tracks_dict.pop(track_id)

            # for track_ids need to do counter
            for track_id in self.ids_do_cnt:
                track_dict_tmp = line_tracks_dict.get(track_id)
                x1 = float(track_dict_tmp.get('last_cx'))
                y1 = float(track_dict_tmp.get('last_cy'))
                x2 = float(track_dict_tmp.get('current_cx'))
                y2 = float(track_dict_tmp.get('current_cy'))

                line_traj = [x1, y1, x2, y2]
                tmp = 0

                for j in range(1, len(line) - 2, 2):
                    cnt_enter, cnt_exit = self.line_count(line_traj, line[j:j + 4])
                    if track_dict_tmp.get('confirm_frames') > self.trust_frames:
                        if cnt_enter:
                            # print track_id
                            tmp += 1
                        elif cnt_exit:
                            # print track_id
                            tmp -= 1
                        else:
                            unconfirmed_enter_frames = track_dict_tmp.get('unconfirmed_enter_frame_idxs')
                            unconfirmed_exit_frames = track_dict_tmp['unconfirmed_exit_frame_idxs']
                            if len(unconfirmed_enter_frames) - len(unconfirmed_exit_frames) > 0:
                                tmp += 1
                            elif len(unconfirmed_enter_frames) - len(unconfirmed_exit_frames) < 0:
                                tmp -= 1
                            track_dict_tmp.update(
                                {
                                    'unconfirmed_enter_frame_idxs': [],
                                    'unconfirmed_exit_frame_idxs': [],
                                })
                    else:
                        if cnt_enter:
                            track_dict_tmp['unconfirmed_enter_frame_idxs'] += [self.frame_idx]
                        elif cnt_exit:
                            track_dict_tmp['unconfirmed_exit_frame_idxs'] += [self.frame_idx]

                if tmp > 0:
                    # self.cnt[line_id]['enter'] += 1
                    track_dict_tmp['confirm_frames'] = -self.ignore_frames
                    track_dict_tmp['enter_frame_idxs'] += [self.frame_idx]
                    line_crossid_dict['enter_ids'] += [track_id]
                    self.cross_result.append([int(track_id), line_id, self.frame_idx, 1])
                    # add for counting time
                    if working_now:
                        self.cnt[line_id]['enter'] += 1
                        self.end_work_flag = True

                elif tmp < 0:
                    self.cnt[line_id]['exit'] += 1
                    track_dict_tmp['confirm_frames'] = -self.ignore_frames
                    track_dict_tmp['exit_frame_idxs'] += [self.frame_idx]
                    line_crossid_dict['exit_ids'] += [track_id]
                    self.cross_result.append([int(track_id), line_id, self.frame_idx, -1]) 
            self.cross_ids.update({line_id: line_crossid_dict})
        pass  # for line, line_id in zip(self.lines, self.line_ids):

    @staticmethod
    def line_count(line1, line2):
        # line1 = [x1,y1,x2,y2] is two points continues trajectory
        # line2 = [x1,y1,x2,y2] is two points of counting line
        """
        A:(line1_x1,line1_y1)   B:(line1_x2,line1_y2)  C:(line2_x1,line2_y1)   D:(line2_x2,line2_y2)
        AB:(line1_x2-line1_x1, line1_y2-line1_y1)   CD: line2_x2-line2_x1, line2_y2-line2_y1)
        """
        line1_x1 = float(line1[0])
        line1_y1 = float(line1[1])
        line1_x2 = float(line1[2])
        line1_y2 = float(line1[3])
        line2_x1 = float(line2[0])
        line2_y1 = float(line2[1])
        line2_x2 = float(line2[2])
        line2_y2 = float(line2[3])
        cnt_enter = 0
        cnt_exit = 0

        delta = float(line1_x2 - line1_x1) * (line2_y1 - line2_y2) - (line2_x1 - line2_x2) * (line1_y2 - line1_y1)
        if (delta <= 1e-6) and (delta >= -1e-6):
            return cnt_enter, cnt_exit

        beta = float(line2_x1 - line1_x1) * (line2_y1 - line2_y2) - (line2_x1 - line2_x2) * (line2_y1 - line1_y1)
        beta_on_delta = beta / delta
        if (beta_on_delta > 1) or (beta_on_delta < 0):
            return cnt_enter, cnt_exit

        miu = float(line1_x2 - line1_x1) * (line2_y1 - line1_y1) - (line2_x1 - line1_x1) * (line1_y2 - line1_y1)
        miu_on_delat = miu / delta
        if (miu_on_delat > 1) or (miu_on_delat < 0):
            return cnt_enter, cnt_exit

        D2 = (line2_y2 - line2_y1) * line1_x2 + (line2_x1 - line2_x2) * line1_y2 + (
            line2_x2 * line2_y1 - line2_x1 * line2_y2)
        D1 = (line2_y2 - line2_y1) * line1_x1 + (line2_x1 - line2_x2) * line1_y1 + (
            line2_x2 * line2_y1 - line2_x1 * line2_y2)
        if D2 < 0 or D1 > 0:
            cnt_enter = 1
        elif D2 > 0 or D1 < 0:
            cnt_exit = 1
        pass
        return cnt_enter, cnt_exit


# Test unit
if __name__ == "__main__":
    # TODO
    track_output = \
        {
            "10":
                {"5":
                     {"timestamp": "02:22.12",
                      "det_box": [1, 2, 3, 4],
                      "state": 2,
                      "update_times": 0,
                      "track_box": [1, 2, 3, 4],
                      "det_idx": "0",
                      "confident": 0.2
                      }
                 },
        }
    # how to use
    lines = [[1092, 12, 123, 123, 123], [1088, 12, 123, 123, 123]]
    counter = PolylineCounter(lines, trust_frames=3, ignore_frames=10)
    for i in range(10):
        counter.update(True, track_output, frame_idx='5')

    print(counter.get_counter_number())
    print(counter.get_current_crossed_ids())

    pass

    """
    --polyline_ids 1  
    --polyline 0.3818 0.6611 0.6224 0.5200 
    --polygon_ids 10 11 12 
    --polygon 0.2010 0.4574 0.2479 0.3713 0.3458 0.3972 0.3599 0.5722 0.2750 0.6598 0.2021 0.5787 
    --polygon 0.3135 0.4056 0.5062 0.3222 0.6307 0.4565 0.3995 0.6176 
    --polygon 0.4911 0.2926 0.5536 0.2241 0.6448 0.2426 0.6833 0.3546 0.6396 0.4704 0.5385 0.4519 
    """
    # # 实例化
    # # parse the plines and pgons: [[2030, 288.0, 865.5120000000001, 1573.44, 207.036]]
    # plines = 1
    # TRUST_FRAMES = 3
    # IGNORE_FRAMES = 20
    # PLCounter = PolylineCounter(polylines=plines, trust_frames=TRUST_FRAMES,
    #                             ignore_frames=IGNORE_FRAMES)
