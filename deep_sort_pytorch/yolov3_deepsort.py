import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results
from utils.counter import counting
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)  # WINDOW_NORMAL 可调整窗口大小
            cv2.resizeWindow("test", args.display_width, args.display_height)  # 后两个参数只支持整数

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        total = 0
        counter = 0
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)
            # print(bbox_xywh)
            # print(cls_conf)
            # print(cls_ids)

            # select person class
            # mask = cls_ids == 0
            # 增加追踪类别
            person = cls_ids == 0
            phone = cls_ids == 67
            mask = person + phone
            # mask = (cls_ids == 20) + (cls_ids == 22)

            bbox_xywh = bbox_xywh[mask]
            # print('**********************')
            # print(bbox_xywh)
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            bbox_xywh[:, 3:] *= 1.2
            # print('**********************')
            # print(bbox_xywh)
            cls_conf = cls_conf[mask]

            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)  # 数组

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                # print(outputs)
                bbox_xyxy = outputs[:, :4]
                # print(bbox_xyxy)
                identities = outputs[:, -1]
                # ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
                ori_im, total = draw_boxes(ori_im, bbox_xyxy, identities)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                ori_im, counter = counting(ori_im, bbox_xyxy)

                results.append((idx_frame - 1, bbox_tlwh, identities))
                # print(results)

            end = time.time()

            if self.args.display:
                phone_num = 0
                # pts = np.array([[[674, 67], [756, 216], [652, 346], [483, 347], [398, 227], [497, 70]]])
                pts = np.array([[[273, 89], [307, 193], [262, 284], [185, 288], [138, 215], [174, 90]]])
                ori_im_copy = ori_im.copy()
                cv2.fillPoly(ori_im_copy, pts, 255)  # 可以填充任意形状
                cv2.addWeighted(ori_im, 0.5, ori_im_copy, 1 - 0.5, 0, ori_im)

                cv2.putText(ori_im, 'Num: ' + str(total), (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(ori_im, 'Phone Track    Time', (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                # cv2.putText(ori_im, str(counter), (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.line(ori_im, (490, 100), (490, 400), (0, 0, 0xFF), 5)

                # 原始
                cv2.imshow("test", ori_im)
                # cv2.waitKey(1)
                if cv2.waitKey(60 * 2) & 0xFF == ord('q'):
                    break

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            write_results(self.save_results_path, results, 'mot')

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--VIDEO_PATH", type=str, default="/dev/video0")
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    # parser.add_argument("--display", action="store_true")
    parser.add_argument("--display", action="store_true", default="--display")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    # parser.add_argument("--display_width", type=int, default=1920)
    # parser.add_argument("--display_height", type=int, default=1080)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    # parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    # print(cfg)
    # print(cfg)
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        """
        cfg:{
        'merge_from_file': <bound method YamlParser.merge_from_file of {...}>, 'merge_from_dict': <bound method YamlParser.merge_from_dict of {...}>, 
        
        'YOLOV3': {'CFG': './detector/YOLOv3/cfg/yolo_v3.cfg', 
        'WEIGHT': './detector/YOLOv3/weight/yolov3.weights', 'CLASS_NAMES': './detector/YOLOv3/cfg/coco.names', 
        'SCORE_THRESH': 0.5, 'NMS_THRESH': 0.4, 
        'merge_from_file': <bound method YamlParser.merge_from_file of {...}>, 'merge_from_dict': <bound method YamlParser.merge_from_dict of {...}>}, 
        
        'DEEPSORT': {'REID_CKPT': './deep_sort/deep/checkpoint/ckpt.t7', 'MAX_DIST': 0.2, 'MIN_CONFIDENCE': 0.3, 
        'NMS_MAX_OVERLAP': 0.5, 'MAX_IOU_DISTANCE': 0.7, 'MAX_AGE': 70, 'N_INIT': 3, 'NN_BUDGET': 100, 
        'merge_from_file': <bound method YamlParser.merge_from_file of {...}>, 'merge_from_dict': <bound method YamlParser.merge_from_dict of {...}>}
        }
        
        args:Namespace(VIDEO_PATH='/dev/video0', cam=0, config_deepsort='./configs/deep_sort.yaml', config_detection='./configs/yolov3.yaml', 
        display='--display', display_height=600, display_width=800, frame_interval=1, save_path='./output/', use_cuda=True)
        """
        # print(cfg)
        # print(args)
        # print(args.display)
        vdo_trk.run()
