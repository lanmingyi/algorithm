import math
import random
import string

random.seed(0)


def rand(a, b):
    return (b-a) * random.random() + a


def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


def sigmoid(x):
    return math.tanh(x)


def dsigmoid(y):
    return y-y**2


class neuralNetwork:
    """三层BP网络"""
    def __init__(self, ni, nh, no):
        # 输入层 隐藏层 输出层
        self.ni = ni + 1
        self.nh = nh
        self.no = no

        # 激活神经网络的所有结点（向量）
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # 权重矩阵
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)

        # 设置随机值
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)

        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # 最后建立动量因子（矩阵）
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError('与输入层节点数不府！')

        # 激活输入层
        for i in range(self.ni-1):
            self.ai[i] = inputs[i]

        # 激活隐藏层
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # 激活输出层
        for k in range(self.no):
            sum = 0.0
            for j in range(self.no):
                sum = 0.0
                for j in range(self.nh):
                    sum = sum + self.ah[j] * self.wo[j][k]
                self.ao[k] = sigmoid(sum)

            return self.ao[:]

    def backPropagate(self, targets, N, M):
        """反向传播"""
        if len(targets) != self.no:
            raise ValueError('与输出节点个数不符！')

        # 计算输出层的误差
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # 计算隐藏层的误差
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # 更新输出层权重
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change

        # 更新输入层权重
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # 计算误差
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k] - self.ao[k])**2
        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('输入层权重：')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('输出层权重：')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, epoch=10000, N=0.5, M=0.1):
        # N:学习速率（learning rate）
        # M:动量因子（momentum factor）
        for i in range(epoch):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('误差 % -.5f' % error)


def demo():
    pat = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]],
    ]
    n = neuralNetwork(2, 2, 1)
    n.train(pat)
    n.test(pat)
    n.weights()


if __name__ == '__main__':
    demo()