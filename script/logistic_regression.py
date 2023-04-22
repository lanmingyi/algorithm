import numpy as np
import os


def loadData(dir):
    trainfileList = os.listdir(dir)
    m = len(trainfileList)
    dataArray = np.zeros((m, 1024))
    labelArray = np.ones((m, 1))

    for i in range(m):
        returnArray = np.zeros((1, 1024))
        fileName = trainfileList[i]
        fr = open('%s/%s' % (dir, fileName))

        for j in range(32):
            lineStr = fr.readline()
            for k in range(32):
                returnArray[0, 32 * j + k] = int(lineStr[k])
        dataArray[i, :] = returnArray

        label = fileName.split('_')[0]
        labelArray[i] = int(label)
    return dataArray, labelArray


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def train(dataArray, labelArray, learning_rate, epoch):
    dataMat = np.mat(dataArray)
    labelMat = np.mat(labelArray)

    m, n = np.shape(dataMat)
    print(m, n)
    weight = np.ones((n, 1))

    for i in range(epoch):
        h = sigmoid(dataMat * weight)

        error = labelMat - h

        weight = weight + learning_rate * dataMat.transpose() * error
    return weight


def test(testdir, weight):
    dataArray, labelArray = loadData(testdir)
    dataMat = np.mat(dataArray)
    labelMat = np.mat(labelArray)

    h = sigmoid(dataMat * weight)
    m = len(h)
    error = 0.

    for i in range(m):
        if int(h[i]) > 0.5:
            print(int(labelMat[i]), 'is classfied as: 1')
            if int(labelMat[i] != 1):
                error += 1
        else:
            print(int(labelMat[i]), 'is classfied as: 0')
            if int(labelMat[i] != 0):
                error += 1

    print(1.0 * (m - error) / m)


data, label = loadData('./train')
weight = train(data, label, learning_rate=0.07, epoch=1000)
test('./test', weight)