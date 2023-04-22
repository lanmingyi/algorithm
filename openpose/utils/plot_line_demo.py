# import cv2
# import numpy as np
#
# img = cv2.imread("test.jpeg")
# # cv2.line(img, (0, 200), (640, 200), (0, 0xFF, 0), 5)
# # cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 5)
#
# # pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
# # pts = [(674, 67), (756, 216), (652, 346), (483, 347), (398, 227), (497, 70)]
# pts = np.array([[[674, 67], [756, 216], [652, 346], [483, 347], [398, 227], [497, 70]]])
# # pts = pts.reshape((-1, 1, 2))
#
# img_copy = img.copy()
# # img = cv2.polylines(img, [pts], True, (0, 255, 0))
# # cv2.polylines(img, pts, True, (0, 255, 0))
# cv2.fillPoly(img_copy, pts, 255)  # 可以填充任意形状
# # cv2.fillConvexPoly(img_copy, pts, 255)  # 只能用来填充凸多边形
# cv2.addWeighted(img, 0.5, img_copy, 1 - 0.5, 0, img)
# # cv2.imwrite('plot_1.jpg', img)
# cv2.imshow('1', img)
# cv2.waitKey()


# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# a = np.array([[[10,10], [100,10], [100,100], [10,100]]], dtype = np.int32)
# b = np.array([[[100,100], [200,230], [150,200], [100,220]]], dtype = np.int32)
# print(a.shape)
# im = np.zeros([240, 320], dtype = np.uint8)
# cv2.polylines(im, a, 1, 255) # 绘制多边形
# cv2.fillPoly(im, b, 255) # 绘制多边形 填充
# plt.imshow(im)
# plt.show()

# 主要利用了opencv的鼠标回调函数：
# 1.点击鼠标左键，画出点
# 2.点击鼠标右键，生成闭合多边形，并输出坐标点。
# 3.点击鼠标中键，删除多边形。
# 4.按Q键，退出。

import cv2
import time
import numpy as np
tpPointsChoose = []
drawing = False
tempFlag = False


def draw_ROI(event, x, y, flags, param):
    global point1, tpPointsChoose,pts,drawing, tempFlag
    if event == cv2.EVENT_LBUTTONDOWN:
        tempFlag = True
        drawing = False
        point1 = (x, y)
        tpPointsChoose.append((x, y))  # 用于画点
    if event == cv2.EVENT_RBUTTONDOWN:
        tempFlag = True
        drawing = True
        pts = np.array([tpPointsChoose], np.int32)
        pts1 = tpPointsChoose[1:len(tpPointsChoose)]
        print(pts1)
    if event == cv2.EVENT_MBUTTONDOWN:
        tempFlag = False
        drawing = True
        tpPointsChoose = []


cv2.namedWindow('video', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('video', draw_ROI)
# cap = cv2.VideoCapture('1.avi')  # 文件名及格式
# cap = cv2.VideoCapture(0)  # 摄像头
# cap = cv2.VideoCapture("rtsp://admin:admin123@192.168.1.2:554/h264/ch1/main/av_stream")
cap = cv2.VideoCapture("../data/test1.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps: {}\nsize: {}".format(fps,size))
vfps = 0.7/fps  # 延迟播放用，根据运算能力调整
while (True):
    # capture frame-by-frame
    ret, frame = cap.read()
    # display the resulting frame
    if (tempFlag == True and drawing == False) :  # 鼠标点击
        cv2.circle(frame, point1, 5, (0, 255, 0), 2)
        for i in range(len(tpPointsChoose) - 1):
            cv2.line(frame, tpPointsChoose[i], tpPointsChoose[i + 1], (255, 0, 0), 2)
    if (tempFlag == True and drawing == True):  #鼠标右击
        cv2.polylines(frame, [pts], True, (0, 0, 255), thickness=2)
    if (tempFlag == False and drawing == True):  # 鼠标中键
        for i in range(len(tpPointsChoose) - 1):
            cv2.line(frame, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)
    time.sleep(vfps)
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
        break
# when everything done , release the capture
cap.release()
cv2.destroyAllWindows()
