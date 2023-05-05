import cv2


def predict(dataset, model, ext):
    global img_y
    x = dataset[0].replace('\\', '/')
    file_name = dataset[1]
    x = cv2.imread(x)
    img_y, image_info = model.detect(x)
    print('img_y', img_y)
    print('image_info', image_info)
    # if cv2.imwrite('./tmp/draw/{}.{}'.format(file_name, ext), img_y):
    #     raise Exception('保存图片时出错.Error saving thepicture.')
    # return image_info
    if cv2.imwrite('object_detection/results/{}.{}'.format(file_name, ext), img_y):
        return image_info
    else:
        raise Exception('保存图片时出错.Error saving thepicture.')
