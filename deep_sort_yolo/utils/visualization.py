import colorsys
import os, sys
import cv2
import numpy as np
import time
# from .ft2 import PutChineseText
from PIL import ImageFont, Image, ImageDraw
# from src.application_util import processing
import processing


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

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255 * r), int(255 * g), int(255 * b)


def rectangle(image, xmin, ymin, xmax, ymax, thickness, color=(0, 0, 0), label=None):
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)

    cv2.rectangle(image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=color, thickness=thickness)

    if label is not None:
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
            if (int(track_data['state'])) != 2 or (int(track_data['update_times'])) > 0:
                # if (int(track_data['state'])) != 2 :
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
    {‘trackId’ : {‘frameIdx’ :
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


def draw_detections(image, dets, color=(0, 255, 0)):
    # font = ImageFont.truetype("data/simhei.ttf", 16)
    # ==================ft===================
    # counting_ft = PutChineseText("data/simhei.ttf")
    # text_ft_size = 12

    for det in dets:
        xmin = int(det['left'])
        ymin = int(det['top'])
        xmax = int(det['right'])
        ymax = int(det['bottom'])
        category = det['name']

        if xmin >= 1820 or ymin > 1000:
            return image
        cv2.rectangle(image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=color, thickness=2)
        pad_score = "%.2f" % (det['confident'])
        # cv2.rectangle(image, pt1=(xmin, ymin), pt2=(xmax, ymin + 16), color=color, thickness=2)
        cv2.putText(image, text=category + ": " + str(pad_score), org=(xmin + 2, ymin + 11), fontFace=0,
                    fontScale=0.4, thickness=1, color=color)

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
        # if category == "person":
        #     label = "员工" +  ":" + str(pad_score)
        # elif category == "DianJi":
        #     label = "电机" +  ":" + str(pad_score)
        # elif category == "JieChuPan":
        #     label = "接触盘" +  ":" + str(pad_score)
        # else:
        #     label = "其他"

        # if(xmin >=1820 or ymin >1000):
        #     return image

        # image = counting_ft.draw_text(image, (int(xmin + 1), int(ymin + 1)), label, text_ft_size, color)
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
                r, g, b = create_unique_color_uchar(int(track_id))
                xmin = int(track_box[0])
                ymin = int(track_box[1])
                xmax = int(track_box[2])
                ymax = int(track_box[3])

                center_x = int((xmin + xmax) / 2.0)
                center_y = int((ymin + ymax) / 2.0)

                color = (b, g, r)
                thickness = -1
                radius = 5
                cv2.circle(image, (center_x, center_y), radius, color, thickness)
                """
                circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> img
                Draws a circle.
                The function circle draws a simple or filled circle with a given center and radius.
                img - Image where the circle is drawn.
                center - Center of the circle.
                radius - Radius of the circle.
                color - Circle color.
                thickness - Thickness of the circle outline, if positive. Negative thickness means that a filled circle is to be drawn.
                lineType - Type of the circle boundary. See the line description.
                shift - Number of fractional bits in the coordinates of the center and in the radius value.
                """
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


def draw_polylines(image, polylines):
    # polylines = [[polyline_id, x, y, x, y, x, y], ...]
    colors = [[255, 0, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]
    color_idx = 0

    for polyline in polylines:
        len_polyline = len(polyline)

        for i in range(1, len_polyline - 2, 2):
            point1 = (int(polyline[i]), int(polyline[i + 1]))
            point2 = (int(polyline[i + 2]), int(polyline[i + 3]))

            # cv2.line(image, point1, point2, colors[int(color_idx % 5)], 2)
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


def draw_working_time(image, TMCounter, end_work_flag, frame_idx, rtsp):
    label_end = ""
    global label_during
    font = ImageFont.truetype("data/simhei.ttf", 42)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    if TMCounter.working_now:
        # ======================chinese====================================
        label = "开始时间: " + TMCounter.start_work_time
        left, top = 30, 30
        draw.text((int(left), int(top)), label, tuple([0, 0, 255]), font=font)

        # #system time
        if rtsp == "Yes":
            now_time = time.strftime("%H:%M:%S", time.localtime())
            time_diff_str = processing.compute_time(TMCounter.start_work_time, now_time)  # time diff with start time
        else:
            time_diff = (frame_idx - TMCounter.start_work_frame_idx) / float(TMCounter.fps)
            time_diff_str = processing.sec2hms(time_diff)

        label_end = "正在工作: " + time_diff_str
        left, top = 700, 30
        draw.text((int(left), int(top)), label_end, tuple([0, 0, 255]), font=font)

    if end_work_flag:
        # ====================chinese============================
        TMCounter.show_last_working_time = True
        if rtsp == "Yes":
            total_time_str, total_time_sec = TMCounter.get_start_end_work_time_rtsp()
        else:
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


def draw_start_end_time(image, camera_id, TMCounter, end_work_flag, frame_idx, rtsp):
    font = ImageFont.truetype("data/simhei.ttf", 42)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    if TMCounter.working_now:
        # ======================chinese====================================
        label_camera = "设备id: " + str(camera_id)
        left, top = 30, 30
        draw.text((int(left), int(top)), label_camera, tuple([0, 0, 255]), font=font)

        label_start = "开始时间: " + TMCounter.start_work_time
        left, top = 700, 30
        draw.text((int(left), int(top)), label_start, tuple([0, 0, 255]), font=font)

        label_working_time = "工时统计: " + "00:00:00"
        left, top = 1400, 30
        draw.text((int(left), int(top)), label_working_time, tuple([0, 0, 255]), font=font)

    if end_work_flag:
        # ====================chinese============================
        label_camera = "设备id: " + str(camera_id)
        left, top = 30, 30
        draw.text((int(left), int(top)), label_camera, tuple([0, 0, 255]), font=font)

        label_end = "结束时间: " + TMCounter.end_work_time
        left, top = 700, 30
        draw.text((int(left), int(top)), label_end, tuple([0, 0, 255]), font=font)
        total_time_str, total_time_sec = TMCounter.get_start_end_work_time_rtsp()
        label_working_time = "工时统计: " + total_time_str
        left, top = 1400, 30
        draw.text((int(left), int(top)), label_working_time, tuple([0, 0, 255]), font=font)

    image = np.asarray(image)
    return image


def draw_time_panel(image_cpu, TMCounter, polylines, PLCounter, end_work_flag, min_time, max_time, rtsp):
    # add background image
    xmin, ymin = 1400, 100  # left top
    x_width, y_height = 410, 500
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # red, blue, green
    polygon_list = [[xmin, ymin], [xmin + x_width, ymin], [xmin + x_width, ymin + y_height], [xmin, ymin + y_height]]
    image_copy = image_cpu.copy()
    cv2.fillConvexPoly(image_copy, np.array(polygon_list), [255, 255, 255])  # 18, 153, 255
    cv2.addWeighted(image_cpu, 0.1, image_copy, 1 - 0.1, 0.0, image_cpu)

    # head
    occupancy_ft = PutChineseText("data/simhei.ttf")  # data/kaiti.ttf
    text_ft_size = 21
    label = "\t操作次数\t|\t开始时间\t|\t工作时长"
    image_cpu = occupancy_ft.draw_text(image_cpu, (xmin + 1, ymin + 10), label, text_ft_size, colors[1])  # 0, 139, 0

    # dict to save the time information
    line_id = polylines[0][0]  # first line
    margin = 50  # head to first line
    limit_item_number = 10

    if end_work_flag:
        op_num = str(PLCounter.get_counter_number().get(line_id).get("enter"))
        start_time = TMCounter.start_work_time
        if rtsp == "Yes":
            work_time_str, work_time_sec = TMCounter.get_start_end_work_time_rtsp()
        else:
            work_time_str, work_time_sec = TMCounter.get_start_end_work_time()
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


# insert_time2db(rtsp, camera_id, min_time, max_time, TMCounter, end_work_flag, db, post_cfg, time_files_path)
def insert_time2db(rtsp, camera_id, min_time, max_time, TMCounter, end_work_flag, db, post_cfg, time_files_path):
    # insert the database only for rtsp
    if rtsp == "Yes" and end_work_flag:
        camera_id = str(camera_id)
        # =====================================================================================
        work_time_str, work_time_sec = TMCounter.get_start_end_work_time_rtsp()
        if work_time_sec > max_time or work_time_sec < min_time:
            is_alarm = "1"
        else:
            is_alarm = "0"
        year_month_day = time.strftime("%Y-%m-%d", time.localtime())
        start_time = year_month_day + " " + TMCounter.start_work_time
        end_time = year_month_day + " " + TMCounter.end_work_time

        # ===========================================================================================
        try:
            select_station_id_sql = ''' SELECT /*+MAX_EXECUTION_TIME(20000)*/ procedure_station_id FROM vm_camera 
                                    WHERE id={}'''.format(camera_id)
            select_station_id = db.select(select_station_id_sql)
            select_station_id = select_station_id[0][0]

        except Exception as e:
            select_station_id = -1
            db.log.error("select_station_id_sql error")
            print("select_station_id_sql error")

        try:
            select_part_id_sql = ''' SELECT /*+MAX_EXECUTION_TIME(20000)*/ procedure_parts_id FROM procedure_station
                                WHERE id={}'''.format(select_station_id)
            select_part_id = db.select(select_part_id_sql)
            select_part_id = select_part_id[0][0]

        except Exception as e:
            select_part_id = -1
            print("select_part_id_sql error")
            db.log.error("select_part_id_sql error")

        # =============================================================================

        insert_time_sql = ''' INSERT INTO procedure_records (camera_id, is_alarm, start_time, end_time, diff_time_seconds, 
                        procedure_station_id, procedure_parts_id) 
                        VALUES ('{camera_id}', '{is_alarm}', '{start_time}', '{end_time}', '{diff_time_seconds}', '{procedure_station_id}', '{procudre_parts_id}')'''.format(
            camera_id=camera_id, is_alarm=is_alarm, start_time=start_time, end_time=end_time,
            diff_time_seconds=work_time_sec, procedure_station_id=select_station_id,
            procudre_parts_id=select_part_id)

        db.insert(insert_time_sql)

        select_data_sql = ''' SELECT /*+MAX_EXECUTION_TIME(20000)*/ id FROM procedure_records  
                            WHERE camera_id={}  and start_time='{}' and end_time='{}' 
                            ORDER BY  id DESC LIMIT 1'''.format(camera_id, start_time, end_time)
        insert_data_id = db.select(select_data_sql)
        # print("insert_data_id is {}".format(insert_data_id))

        # insert the data_id   #haier_dashboard.linkdome.cn
        # https://haier_dashboard.linkdome.cn/api/notify/info?debug=1&
        try:
            os.system("curl -d \"id={}\" {}".format(insert_data_id[0][0], post_cfg))
        except Exception as e:
            print("curl post send error")
            db.log.error("curl post send error")

        # create the start and end files
        ROOT_DIR = os.path.join(time_files_path, camera_id)
        if not os.path.exists(ROOT_DIR):
            os.makedirs(ROOT_DIR)
        start_time_form = "-".join(TMCounter.start_work_time.split(":"))
        end_time_form = "-".join(TMCounter.end_work_time.split(":"))
        START_FILE_NAME = "start_" + camera_id + "_" + year_month_day + "_" + start_time_form + ".txt"
        END_FILE_NAME = "end_" + camera_id + "_" + year_month_day + "_" + end_time_form + ".txt"
        os.mknod(os.path.join(ROOT_DIR, START_FILE_NAME))
        os.mknod(os.path.join(ROOT_DIR, END_FILE_NAME))
        # open(os.path.join(ROOT_DIR, START_FILE_NAME), mode='wb', buffering=0).close()
        # open(os.path.join(ROOT_DIR, END_FILE_NAME), mode='wb', buffering=0).close()
    else:
        pass


def save_image(image_cpu, camera_id, TMCounter, image_files_path):
    camera_id = str(camera_id)
    year_month_day = time.strftime("%Y-%m-%d", time.localtime())
    ROOT_DIR = os.path.join(os.path.join(image_files_path, camera_id), year_month_day)
    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)
    if TMCounter.save_start_img:
        start_time_form = "-".join(TMCounter.start_work_time.split(":"))
        START_FILE_NAME = "start_" + camera_id + "_" + year_month_day + "_" + start_time_form + ".jpg"
        cv2.imwrite(os.path.join(ROOT_DIR, START_FILE_NAME), image_cpu, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        TMCounter.save_start_img = False

    if TMCounter.save_end_img:
        end_time_form = "-".join(TMCounter.end_work_time.split(":"))
        END_FILE_NAME = "end_" + camera_id + "_" + year_month_day + "_" + end_time_form + ".jpg"
        cv2.imwrite(os.path.join(ROOT_DIR, END_FILE_NAME), image_cpu, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        TMCounter.save_end_img = False
    pass


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

        id = int(polygon[0])

        if id == 10 or id == 12:
            color_idx = 0
        if id == 11:
            color_idx = 1

        image_copy = image.copy()
        cv2.fillConvexPoly(image_copy, np.array(polygon_list), colors[int(color_idx % 5)])
        cv2.addWeighted(image, 0.6, image_copy, 1 - 0.6, 0.0, image)
        color_idx += 1
    pass


def draw_occupancy(image, polygons, counter_output):
    # polygons = [[polyline_id, x, y, x, y, x, y], ...]
    colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [255, 0, 255]]
    color_idx = 0
    # =============ft=================
    # counting_ft = PutChineseText("data/simhei.ttf")
    # text_ft_size = 22

    # ================imagefont=======================
    font = ImageFont.truetype("data/simhei.ttf", 20)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    # =================================================
    polygons_id = []

    for polygon in polygons:
        if len(polygon) < 7:
            continue
        polygons_id.append(polygon[0])
    pass

    for polygon in polygons:
        if len(polygon) < 7:
            continue
        pass
        # ##=================english===========================
        # pt1 = (int(polygon[1]), int(polygon[2]))
        # pt2 = (int(polygon[3]), int(polygon[4]))
        # id = int(polygon[0])
        # label = "occupancy: " + str(counter_output.get(id))
        # text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)
        # left = pt1[0] + (pt2[0] - pt1[0]) / 2 - text_size[0][0] / 2
        # top = pt1[1] + (pt2[1] - pt1[1]) / 2 - text_size[0][1] / 2
        # cv2.putText(image, label, (int(left), int(top)), cv2.FONT_HERSHEY_COMPLEX, 0.6, colors[int(color_idx % 5)], 1)
        # color_idx += 1

        # ##=================chinese===========================
        pt1 = (int(polygon[1]), int(polygon[2]))
        pt2 = (int(polygon[3]), int(polygon[4]))
        left = pt1[0] + (pt2[0] - pt1[0]) / 2
        top = pt1[1] + (pt2[1] - pt1[1]) / 2
        id = int(polygon[0])
        label = ""
        color = tuple(colors[int(color_idx % 5)])

        assert len(polygons_id) == 3, "error polygons number at draw_occupancy function"
        if id == polygons_id[0] or id == polygons_id[2]:
            label = "人数: " + str(counter_output.get(id))
            color_idx = 0
            color = tuple(colors[int(color_idx % 5)])
        if id == polygons_id[1]:
            color_idx = 0
            label = "电机数: " + str(counter_output.get(id))
            color = tuple(colors[int(color_idx % 5)])
        # =======================ft======================
        # image = counting_ft.draw_text(image, (int(left), int(top)), label, text_ft_size, colors[int(color_idx % 5)])

        # ================imagefont=======================
        # font = ImageFont.truetype("data/simhei.ttf", 20)
        # image = Image.fromarray(image)
        # draw = ImageDraw.Draw(image)
        draw.text((int(left), int(top)), label, color, font=font)
    image = np.asarray(image)
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
    logo_path = "data/lingtu_logo.png"
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
