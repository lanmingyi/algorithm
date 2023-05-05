from object_detection.core import process, predict


def c_main(path, model, ext):
    image_data = process.pre_process(path)

    image_info = predict.predict(image_data, model, ext)
    # print('image_info', image_info)

    return image_data[1] + '.' + ext, image_info


if __name__ == '__main__':
    pass
