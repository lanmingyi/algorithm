import os
import cv2
import numpy as np

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from detector.da_unet.model import da_u_net
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT_DIR = FILE.parents[2]  # root directory
checkpoint_dir = os.path.join(ROOT_DIR, "models/da_unet")
RESULT_PATH = os.path.join(ROOT_DIR, "results")


def run(image_path):
    batch_size = 1
    size = 512
    over_lap = 256
    img = tf.placeholder(tf.float32, [batch_size, size, size, 3])
    pred = tf.nn.sigmoid(da_u_net(img, is_training=False))
    print('pred', pred)
    saver = tf.train.Saver(var_list=tf.global_variables())
    with tf.Session() as sess:
        print('sess', sess)
        RESULT_NAME = image_path.split('/')[-1].split('.')[0] + '.jpg'
        save_path = os.path.join(RESULT_PATH, RESULT_NAME)

        tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print('ckpt', ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            print('ckpt1', ckpt)
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print('Loading Checkpoint Success !')

        ori_image = np.uint8(cv2.imread(image_path))
        # img1 = cv2.imread(image_path)
        # img1 = cv2.resize(img1,(512, 512))
        # ori_image = np.uint8(img1)
        ori_image = ori_image[..., 0:3]
        print('Start to cutting {}'.format(image_path))
        image_list = []
        predict_list = []
        oh, ow = ori_image.shape[0], ori_image.shape[1]
        dert = size-over_lap
        h_n = (oh - size) // dert
        h_step = h_n+1
        w_n = (ow - size) // dert
        w_step = w_n+1
        h_rest = oh-size-h_n*dert
        w_rest = ow-size-w_n*dert

        for h in range(h_step):
            for w in range(w_step):
                image_sample = ori_image[(h * dert):(h * dert + size), (w * dert):(w * dert + size), :]
                image_list.append(image_sample)
            image_list.append(ori_image[(h * dert):(h * dert + size), -size:, :])
        for w in range(w_step):
            image_list.append(ori_image[-size:, (w * dert):(w * dert + size), :])
        image_list.append(ori_image[-size:, -size:, :])
        print(len(image_list))

        print('Start to predict {}'.format(image_path))

        for x_batch1 in image_list:
            x_batch = np.expand_dims(x_batch1, axis=0) / 255.0
            predict = sess.run(pred, feed_dict={img: x_batch})
            predict = np.squeeze(predict)
            predict[predict > 0.5] = 1
            predict[predict <= 0.5] = 0
            predict_list.append(predict)
        count_temp = 0
        tmp = np.ones([ori_image.shape[0], ori_image.shape[1]]).astype(np.uint8)
        big_w = size - over_lap // 2
        small_w = over_lap//2
        print('Start to cut {}'.format(image_path))
        for h in range(h_step):
            for w in range(w_step):
                if h == 0 or h == h_step-1:
                    dh = size-over_lap//2
                else:
                    dh = dert
                if w == 0 or w == w_step-1:
                    dw = size-over_lap//2
                else:
                    dw = dert

                tmp[(h-1) * over_lap+(big_w if h > 0 else over_lap):(h-1) * over_lap+(big_w if h > 0 else over_lap)+dh, (w - 1)*over_lap+(big_w if w > 0 else over_lap):(w-1)*over_lap+(big_w if w > 0 else over_lap)+dw] \
                    = predict_list[count_temp][small_w*int(h > 0):dh+small_w*int(h > 0), small_w*int(w > 0):dw+small_w*int(w > 0)]
                count_temp += 1
            tmp[(h-1) * over_lap + (big_w if h > 0 else over_lap):(h-1) * over_lap + (big_w if h > 0 else over_lap)+dh, -w_rest:] \
                = predict_list[count_temp][small_w if h > 0 else 0:(small_w if h > 0 else 0)+dh, -w_rest:]
            count_temp += 1
        for w in range(w_step):
            tmp[-h_rest:, (w-1) * over_lap + (big_w if h > 0 else over_lap):(w-1) * over_lap + (big_w if h > 0 else over_lap)+dw] \
                = predict_list[count_temp][-h_rest:, small_w if w > 0 else 0:+(small_w if w > 0 else 0)+dw]
            count_temp += 1
        tmp[-h_rest:, -w_rest:] = predict_list[count_temp][-h_rest:, -w_rest:]

        tmp[tmp == 1] = 255
        print('tmp', tmp)
        print('Save {}'.format(save_path))
        cv2.imwrite(save_path, tmp)
        sess.close()

    return save_path


if __name__ == '__main__':
    image_path = os.path.join('D:/usr/server/upload/', 'temp/test3.tif')
    run(image_path)
