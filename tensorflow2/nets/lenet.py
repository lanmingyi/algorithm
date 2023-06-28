from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import input_data
import numpy
from six.moves import urllib

mnist = input_data.read_data_sets('../../dataset/MNIST/raw', one_hot=True)


# 初始化操作，权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置初始化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积操作
def conv2d(x, W, argument):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=argument)


# 最大池化操作
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 初始化操作
sess = tf.InteractiveSession()  # 开启流程
x = tf.placeholder("float", shape=[None, 784])  # 输入，None占位符，先把流程走下去，后面再确定
y_ = tf.placeholder("float", shape=[None, 10])  # 输出

# First convolutional layer
W_conv1 = weight_variable([5, 5, 1, 6])
b_conv1 = bias_variable([6])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 'SAME') + b_conv1)  # same: 输入图像自动补零使得输出的特征图维度和输入时一样的
h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer
W_conv2 = weight_variable([5, 5, 6, 16])
b_conv2 = bias_variable([16])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 'VALID') + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Three fully connected layers
W_fc1 = weight_variable([5*5*16, 120])
b_fc1 = bias_variable([120])

h_pool2_flat = tf.reshape(h_pool2, [-1, 5*5*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder("float")  # 保留多少比例的神经元，被赋值为50%
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([120, 84])
b_fc2 = bias_variable([84])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([84, 10])
b_fc3 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

# Evaluation
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))  # 目标函数为交叉熵
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 利用adam优化策略进行训练
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())  # 运行流程
for i in range(12000):  # 12000个batch，整个训练过程总共遍历了训练样本100次
    batch = mnist.train.next_batch(50)  # 每个batch有50个训练样本，总共6万个训练样本，每120个batch遍历所有样本一次
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})  # 正式赋值x和y
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))