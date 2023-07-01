import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

# # Pytorch查看模型网络结构
# # import自己的网络，比如from net import Net，然后vgg=models.vgg19().to(device)改成net = Net().to(device)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# vgg = models.vgg19().to(device)
# summary(vgg, (3, 224, 224))

# # torch.device
# # Via a string:
# torch.device('cuda:0')
# torch.device('cpu')
# torch.device('cuda')  # current cuda device
#
# # Via a string and device ordinal:
# torch.device('cuda', 0)
# torch.device('cpu', 0)


# # torch.ones
# torch.ones(2, 3)
# print(torch.ones(2, 3))
# torch.ones(5)
# print(torch.ones(5))


# # torch.zeros
# print(torch.zeros(2, 3))
# print(torch.zeros(5))


# # torch.cat
# x = torch.randn(2, 3)
# print(x)
# A = torch.cat((x, x, x), 0)
# print(A)
# B = torch.cat((x, x, x), 1)
# print(B)


# # torch.linespace
# A = torch.linspace(3, 10, steps=5)
# print(A)
# B = torch.linspace(-10, 10, steps=5)
# print(B)
# C = torch.linspace(start=-10, end=10, steps=5)
# print(C)
# D = torch.linspace(start=-10, end=10, steps=1)
# print(D)


# # torch.nn.functional.pad
# t4d = torch.empty(3, 3, 4, 2)
# p1d = (1, 1)  # pad lsat dim by 1 on each side
# out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
# print(out.size())
#
# p2d = (1, 1, 2, 2)  # pad last dim by (1, 1) and 2nd to last by (2, 2)
# out = F.pad(t4d, p2d, "constant", 0)
# print(out.size())
#
# p3d = (0, 1, 2, 1, 3, 3)  # pad by (0,1), (2,1), (3,3)
# out = F.pad(t4d, p3d, "constant", 0)
# print(out.size)


# # torch.nn.functional.max_pool2d
# # pool of square window of size=3, stride=2
# m = nn.MaxPool2d(3, stride=2)
# # pool of non-square window
# # m = nn.MaxPool2d((3, 2), stride=(2, 1))
# input = torch.randn(20, 16, 50, 32)
# print(input)
# output = m(input)
# print(output)


# # torch.nn.Conv2d
# # With square kernels and equal stride
# m = nn.Conv2d(16, 33, 3, stride=2)
# # non-square kernels and unequal stride and with padding
# m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# # non-square kernels and unequal stride and with padding and dilation
# m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
# input = torch.randn(20, 16, 50, 100)
# output = m(input)
# print(output)


# # torch.nn.BatchNorm2d
# # With Learnable Parameters
# m = nn.BatchNorm2d(100)
# # Without Learnable Parameters
# m = nn.BatchNorm2d(100, affine=False)
# input = torch.randn(20, 100, 35, 45)
# print(input)
# output = m(input)
# print(output)


# # torch.nn.LeakyReLU
# m = nn.LeakyReLU(0.1)
# input = torch.randn(2)
# print(input)
# output = m(input)
# print(output)


# # torch.nn.ReLU
# m = nn.ReLU()
# input = torch.randn(2)
# output = m(input)
# print(output)
#
# # An implementation of CReLU - https://arxiv.org/abs/1603.05201
# m = nn.ReLU()
# input = torch.randn(2).unsqueeze(0)
# output = torch.cat((m(input), m(-input)))
# print(output)


# # torch.nn.Linear
# m = nn.Linear(20, 30)
# input = torch.randn(128, 20)
# output = m(input)
# print(output)
# print(output.size)


#
