# import tensorflow as tf
# tf.test.is_gpu_available()


# import tensorflow as tf
# print(tf.test.is_built_with_cuda())
# print(tf.config.list_physical_devices('GPU'))
# keras = tf.keras
# model = keras.Sequential()
# print(keras.Sequential())
# print(tf.__version__)
# print(tf.test.is_gpu_available())
# tf.config.list_physical_devices('GPU')

# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import tensorflow as tf
with tf.device('/GPU:0'):
    A = tf.constant([[3, 2], [5, 2]])
    print(tf.eye(2, 2))

# # matplotlib支持的后端
# import matplotlib as mpl
# mpl.use('Qt5Agg')
# print(mpl.get_backend())
# # print(mpl.rcsetup.interactive_bk)
