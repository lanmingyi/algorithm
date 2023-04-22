"""
Copyright (c) 2021 Bright Mind. All rights reserved.
Written by MingYi Lan
"""

import os
import sys
import cv2
import time
from openpose.utils.pose_time_counter import PoseTime
# from openpose.utils import pose_visualization_api
import openpose.utils.pose_visualization_api as pose_visualization
from flask import Blueprint, request, Response
app = Blueprint('openpose', __name__)


try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(dir_path + '/openpose')
    os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/openpose/bin' + ';'
    import pyopenpose as op
    # # Starting Openpose
    # params = {"model_folder": "openpose/models/"}
    # opWrapper = op.WrapperPython()
    # opWrapper.configure(params)
    # opWrapper.start()

except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake '
          'and have this Python script in the right folder?')
    raise e


def pose_system():

    # params = dict()
    # params["model_folder"] = "models/"
    params = {"model_folder": "openpose/models/"}
    video_path = 0
    touch_zones = [
        [600, 350, 900, 350, 900, 600, 600, 600],
        # [1129, 320, 1121, 623, 804, 589, 822, 290],
        # [993, 139, 985, 639, 456, 652, 472, 122]
    ]
    interval = 3
    pose_counter = ''
    max_time = 60
    min_time = 25

    # Starting Openpose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process
    datum = op.Datum()
    # cv2.namedWindow('Pose System 1.0.0 - Python', cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(video_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS), 2)
    pose_time = PoseTime(touch_zones, fps, trust_frames=2)
    frame_idx = -1
    yield b'--frame\r\n'
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1920, 1080))
        frame_idx += 1  # start of frame = 0
        if 0 != (int(frame_idx) % int(interval)):
            continue
        pass

        # frame_show = frame.copy()

        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        keypoints = datum.poseKeypoints
        # print(keypoints)
        # keypoints[:, 4, :]
        # print(keypoints[:, 4, :])
        start_work,  end_work, diff_time_secs = pose_time.is_touch_real(keypoints, frame_idx)
        # print(diff_time_secs)

        end_work = (start_work is True) and (diff_time_secs > 2)
        if end_work is True:
            pose_time.end_work_frame_idx = frame_idx  # end the work
            pose_time.end_work_time = time.strftime("%H:%M:%S", time.localtime())
            pose_time.start_work = False  # reset the flag
            pose_time.end_work = False  # reset the flag
            pose_time.diff_time_secs = 0
            pose_time.op_num += 1

        frame_show = datum.cvOutputData
        frame_show = pose_visualization.draw_Polygons(frame_show, touch_zones)
        frame_show = pose_visualization.draw_top_pose_time(frame_show, pose_time, end_work)
        frame_show = pose_visualization.draw_right_pose_time(frame_show, pose_time, pose_counter,
                                                             end_work, min_time, max_time)

        # cv2.imshow("Pose System 1.0.0 - Python", frame_show)
        # key = cv2.waitKey(1)
        # if key & 255 == 27:  # ESC
        #     break
        ret, frame = cv2.imencode('.jpg', frame_show)
        frame = frame.tobytes()

        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'


@app.route('/openpose')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    # return Response(gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(pose_system(), mimetype='multipart/x-mixed-replace; boundary=frame')
