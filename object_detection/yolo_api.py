import datetime
import logging as rel_log
import os
import shutil
from datetime import timedelta
from flask import *
from object_detection.processor.AIDetector_pytorch import Detector

# import core.main
from object_detection.core.main import *


ALLOWED_EXTENSIONS = set(['png', 'jpg'])
app = Blueprint('objectDetection', __name__)
app.secret_key = 'secret!'

# werkzeug_logger = rel_log.getLogger('werkzeug')
# werkzeug_logger.setLevel(rel_log.ERROR)  #  设置后，启动flask不显示Running on


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/objectDetection', methods=['GET', 'POST'])
def detection_file():
    initialize()
    data = request.json
    image_path = os.path.join('D:/usr/server/upload/', data['filename'])
    pid, image_info = c_main(image_path, Detector(), data['filename'].split('/')[1].split('.')[0])
    if pid:
        return jsonify({'status': 1,
                    'resultUrl': 'http://127.0.0.1:50000/object_detection/results/' + pid,
                    'image_info': image_info})
    else:
        return jsonify({'status': 0})

    # if file and allowed_file(file.filename):
    #     src_path = os.path.join('object_detection/uploads', file.filename)
    #     file.save(src_path)
    #     shutil.copy(src_path, 'object_detection/images')
    #     image_path = os.path.join('object_detection/images', file.filename)
    #     print('image_path', image_path)
    #     print('Detector()', Detector())
    #     print('file.filename', file.filename.rsplit('.', 1)[1])



@app.route("/download", methods=['GET'])
def download_file():
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    return send_from_directory('object_detection/data', 'testfile.zip', as_attachment=True)


# show photo
@app.route('/<path:file>', methods=['GET'])
def show_photo(file):
    if request.method == 'GET':
        if not file is None:
            image_data = open(f'{file}', "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response


def initialize():
    files = ['object_detection/uploads', 'object_detection/images', 'object_detection/results']
    for ff in files:
        if not os.path.exists(ff):
            os.makedirs(ff)
