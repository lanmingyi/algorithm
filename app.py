from importlib import import_module
import os
from flask import Flask, render_template, Response
from flask import *
import flask_cors
import cv2
from datetime import timedelta
from pathlib import Path

from image_processing.views import testview, basicFuncViews, histogramViews, segmentationViews, smoothSharpenViews, \
    repairViews, morphologicalViews, filesViews
from detector import detect_api
# from openpose import pose_api
# from landslide_detection.da_unet import da_unet_api


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
FILE = Path(__file__).resolve()
# print('file', FILE.parents[0])
# print('file', FILE)

# image processing
app.register_blueprint(testview.app)
app.register_blueprint(filesViews.app)
app.register_blueprint(basicFuncViews.app)
app.register_blueprint(histogramViews.app)
app.register_blueprint(segmentationViews.app)
app.register_blueprint(smoothSharpenViews.app)
app.register_blueprint(morphologicalViews.app)
app.register_blueprint(repairViews.app)

# object detection
app.register_blueprint(detect_api.app)

# openpose
# app.register_blueprint(pose_api.app)
# landslide detection
# app.register_blueprint(da_unet_api.app)

# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

# 跨域
flask_cors.CORS(app, resources=r'/*')


# # 添加header解决跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response

# app = Flask(__name__)

# # import camera driver
# if os.environ.get('CAMERA'):
#     Camera = import_module('camera_' + os.environ['CAMERA']).Camera
# else:
#     from camera import Camera
# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera


class VideoCamera(object):
    def __init__(self):
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # 在这里处理视频帧
        cv2.putText(image, "hello", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def gen(camera):
    """Video streaming generator function."""
    yield b'--frame\r\n'
    while True:
        frame = camera.get_frame()
        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    # return Response(gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/')
# def index():
#     """home page."""
#     # return render_template('./index.html')
#     # return redirect(url_for('static', filename='./index.html'))


@app.route('/<path:file>', methods=['GET'])
def show_photo(file):
    if request.method == 'GET':
        if not file is None:
            # print('FILE.parents[0]', FILE.parents[0] / f'{file}')
            image_data = open(FILE.parents[0] / f'{file}', "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response


@app.route("/download", methods=['GET'])
def download_file():
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    # return send_from_directory('object_detection/data', 'testfile.zip', as_attachment=True)
    return send_from_directory('results', 'testfile.zip', as_attachment=True)


def initialize():
    files = ['results', 'landslide_detection/results', 'image_processing/results']
    for ff in files:
        if not os.path.exists(ff):
            os.makedirs(ff)


if __name__ == '__main__':
    initialize()
    app.run(host='0.0.0.0', port=50000, debug=True)
