import os
from flask import *
from detector.yolov5.detect import run as yolov5_detect, parse_opt
from detector.mask_rcnn.prediction import run as mask_run_detect

app = Blueprint('objectDetection', __name__)
# print('name', __name__)
# app.secret_key = 'secret!'
ALLOWED_EXTENSIONS = set(['png', 'jpg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/detect/yolov5', methods=['GET', 'POST'])
def yolov5_detect():
    data = request.json
    image_path = os.path.join('D:/usr/server/upload/temp/', data['filename'])
    opt = parse_opt()
    opt.source = image_path
    # print('opt.project', opt.project)
    result_path = yolov5_detect(**vars(opt))
    # print('result_path.split(/)', result_path.split('\\algorithm')[1].replace('\\', '/'))
    result_path = result_path.split('\\algorithm')[1].replace('\\', '/')
    # print("'http://127.0.0.1:50000/'", 'http://127.0.0.1:50000' + result_path)

    if result_path:
        return jsonify({'status': 1,
                        'resultUrl': 'http://127.0.0.1:50000' + result_path, })
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


@app.route('/detect/maskRcnn', methods=['GET', 'POST'])
def mask_rcnn_detect():
    data = request.json
    image_path = os.path.join('D:/usr/server/upload/temp/', data['filename'])
    # print('opt.project', opt.project)
    result_path = mask_run_detect()
    # print('result_path.split(/)', result_path.split('\\algorithm')[1].replace('\\', '/'))
    result_path = result_path.split('\\algorithm')[1].replace('\\', '/')
    # print("'http://127.0.0.1:50000/'", 'http://127.0.0.1:50000' + result_path)

    if result_path:
        return jsonify({'status': 1,
                        'resultUrl': 'http://127.0.0.1:50000' + result_path, })
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

