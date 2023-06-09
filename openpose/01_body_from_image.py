# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/openpose');
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/openpose/bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python')
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="data/COCO_val2014_000000000338.jpg",
                        help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--video_path", default="rtsp://admin:admin123@192.168.1.2:554/h264/ch1/main/av_stream",
                        help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1]) - 1:
            next_item = args[1][i + 1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    # cap = cv2.VideoCapture(args[0].video_path)
    # ret, imageToProcess = cap.read()
    # print(imageToProcess)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    keypoints = datum.poseKeypoints
    right_hand_list = {}
    for id, keypoints in enumerate(keypoints):
        print(id)
        print(keypoints)
        right_hand = keypoints[4, :-1]
        left_hand = keypoints[7, :-1]
        # right_hand_list[id] = right_hand
        print(keypoints.shape)
        print(right_hand.shape)
        print(keypoints)
        print(right_hand)
        print(left_hand)
        print(right_hand[0])
    # right_hand = keypoints[:, 4, :-1]
    # left_hand = keypoints[:, 7, :-1]
    # print(keypoints.shape)
    # print(right_hand.shape)
    # print(keypoints)
    # print(right_hand)
    # print(left_hand)
    # Display Image
    # print("Body keypoints: \n" + str(datum.poseKeypoints))
    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    cv2.waitKey(0)
except Exception as e:
    print(e)
    sys.exit(-1)
