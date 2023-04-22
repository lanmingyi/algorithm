# encoding=utf-8
import pandas as pd
import numpy as np
import time

# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class KNN(object):

    def __init__(self):
        self.k = 10

    def predict(self, testset, trainset, train_labels):
        predict_ = []
        count = 0
        for test_vec in testset:
            print(count)
            count += 1
            knn_list = []  # 当前k个最近邻居
            max_index = -1  # 当前k个最近邻居中距离最远点的坐标
            max_dist = 0  # 当前k个最近邻居中距离最远点的距离

            # 先将前k个点放入k个最近邻居中，充满knn_list
            for i in range(self.k):
                label = train_labels[i]
                train_vec = trainset[i]

                dist = np.linalg.norm(train_vec - test_vec)

                knn_list.append((dist, label))

            # 剩下的点
            for i in range(self.k, len(train_labels)):
                label = train_labels[i]
                train_vec = trainset[i]

                dist = np.linalg.norm(train_vec - test_vec)

                # 寻找k个邻近点距离最远的点
                if max_index < 0:
                    for j in range(10):
                        if max_dist < knn_list[j][0]:
                            max_index = j
                            max_dist = knn_list[max_index][0]

                if dist < max_dist:
                    knn_list[max_index] = (dist, label)
                    max_index = -1
                    max_dist = 0

            class_total = 10
            class_count = [0 for i in range(class_total)]
            for dist, label in knn_list:
                class_count[label] += 1

            mmax = max(class_count)

            for i in range(class_total):
                if mmax == class_count[i]:
                    predict_.append(i)
                    break
        return np.array(predict_)


if __name__ == '__main__':
    print('Start read data')
    time1 = time.time()
    raw_data = pd.read_csv("../data/train.csv", header=0)
    time2 = time.time()
    print('Read data cost ', time2 - time1, ' second', '\n')

    print(raw_data.info())
    print(raw_data.head())

    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.25, random_state=33)

    print(train_features.shape)
    print(test_features.shape)

    print('Start predicting')
    k = KNN()
    test_predict = k.predict(test_features, train_features, train_labels)
    time3 = time.time()
    print('Predicting cost ', time3 - time2, ' seocnd', '\n')

    score = accuracy_score(test_labels, test_predict)
    print('The accuracy score is ', score)
