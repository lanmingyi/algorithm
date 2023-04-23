import torch
import logging
import numpy as np
import cv2
import sys
# print(sys.path)
from .darknet import Darknet
from .yolo_utils import get_all_boxes, nms, post_process, xywh_to_xyxy, xyxy_to_xywh
from .nms import boxes_nms


class YOLOv3(object):
    """
    __init__()：在实例 (通过 __new__()) 被创建之后，返回调用者之前调用。其参数与传递给类构造器表达式的参数相同。
    一个基类如果有 __init__() 方法，则其所派生的类如果也有 __init__() 方法，就必须显式地调用它以确保实例基类部分的正确初始化；
    例如: super().__init__([args...]).

    因为对象是由 __new__() 和 __init__() 协作构造完成的 (由 __new__() 创建，并由 __init__() 定制)，
    所以 __init__() 返回的值只能是 None，否则会在运行时引发 TypeError。

    __call__()：此方法会在实例作为一个函数被"调用"时被调用；如果定义了此方法，则 x(arg1, arg2, ...) 就大致可以被改写为 type(x).__call__(x, arg1, ...)。

    """
    def __init__(self, cfgfile, weightfile, namesfile, score_thresh=0.7, conf_thresh=0.01, nms_thresh=0.45,
                 is_xywh=False, use_cuda=True):
        # net definition
        self.net = Darknet(cfgfile)
        self.net.load_weights(weightfile)
        logger = logging.getLogger("root.detector")
        logger.info('Loading weights from %s... Done!' % (weightfile))
        self.device = "cuda" if use_cuda else "cpu"
        self.net.eval()
        self.net.to(self.device)

        # constants
        self.size = self.net.width, self.net.height
        self.score_thresh = score_thresh
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.use_cuda = use_cuda
        self.is_xywh = is_xywh
        self.num_classes = self.net.num_classes
        self.class_names = self.load_class_names(namesfile)

    def __call__(self, ori_img):
        # img to tensor
        assert isinstance(ori_img, np.ndarray), "input must be a numpy array!"
        # img = ori_img.astype(np.float) / 255.
        img = ori_img.astype(np.float64) / 255.
        # img = ori_img.astype(float) / 255.  # 或直接使用float

        img = cv2.resize(img, self.size)
        # from_numpy:Creates a Tensor from a numpy.ndarray.
        # permute:Returns a view of the original tensor with its dimensions permuted.
        # unsqueeze: 0是行，1是列
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)

        # forward
        # torch.no_grad():Context-manager that disabled gradient calculation. 禁用梯度计算的上下文管理器
        # 强制之后的内容不进行计算图构建。计算或不计算的结果实际上是没有区别的。
        with torch.no_grad():
            # torch.tensor.to:返回 tensor([[-0.5044,  0.0005],[ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
            img = img.to(self.device)
            out_boxes = self.net(img)
            boxes = get_all_boxes(out_boxes, self.conf_thresh, self.num_classes,
                                  use_cuda=self.use_cuda)  # batch size is 1
            # boxes = nms(boxes, self.nms_thresh)

            boxes = post_process(boxes, self.net.num_classes, self.conf_thresh, self.nms_thresh)[0].cpu()
            boxes = boxes[boxes[:, -2] > self.score_thresh, :]  # bbox xmin ymin xmax ymax

        if len(boxes) == 0:
            bbox = torch.FloatTensor([]).reshape([0, 4])
            cls_conf = torch.FloatTensor([])
            cls_ids = torch.LongTensor([])
        else:
            height, width = ori_img.shape[:2]
            bbox = boxes[:, :4]
            if self.is_xywh:
                # bbox x y w h
                bbox = xyxy_to_xywh(bbox)

            bbox *= torch.FloatTensor([[width, height, width, height]])
            cls_conf = boxes[:, 5]
            cls_ids = boxes[:, 6].long()
        return bbox.numpy(), cls_conf.numpy(), cls_ids.numpy()

    def load_class_names(self, namesfile):
        with open(namesfile, 'r', encoding='utf8') as fp:
            class_names = [line.strip() for line in fp.readlines()]
        return class_names


def demo():
    import os
    from vizer.draw import draw_boxes

    yolo = YOLOv3("cfg/yolo_v3.cfg", "weight/yolov3.weights", "cfg/coco.names")
    print("yolo.size =", yolo.size)
    root = "./demo"
    resdir = os.path.join(root, "results")
    os.makedirs(resdir, exist_ok=True)
    files = [os.path.join(root, file) for file in os.listdir(root) if file.endswith('.jpg')]
    files.sort()
    for filename in files:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox, cls_conf, cls_ids = yolo(img)

        if bbox is not None:
            img = draw_boxes(img, bbox, cls_ids, cls_conf, class_name_map=yolo.class_names)
        # save results
        cv2.imwrite(os.path.join(resdir, os.path.basename(filename)), img[:, :, (2, 1, 0)])
        # imshow
        # cv2.namedWindow("yolo", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("yolo", 600,600)
        # # (2, 1, 0)将图片从cv2.COLOR_BGR2RGB转换成BGR。H,W,C
        # cv2.imshow("yolo",img[:,:,(2,1,0)])
        # cv2.waitKey(0)


if __name__ == "__main__":
    demo()
