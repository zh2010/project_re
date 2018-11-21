# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def tst_single_channel():
    print("conv1d sample")

    a = range(16)

    x = torch.Tensor(a)

    x = x.view(1, 1, 16)

    print("x variable:", x)

    b = torch.ones(3)

    b[0] = 0.1

    b[1] = 0.2

    b[2] = 0.3

    # weights = Variable(b)  # torch.randn(1,1,2,2)) #out_channel*in_channel*H*W

    # weights = weights.view(1, 1, 3)
    weights = b.view(1, 1, 3)

    print("weights:", weights)

    y = F.conv1d(x, weights, padding=0)

    print("y:", y)


def tst_multi_channel():
    print("conv1d sample")

    a = range(16)

    x = torch.Tensor(a)

    x = x.view(1, 2, 8)

    print("x variable:", x)

    b = torch.ones(6)

    b[0] = 0.1

    b[1] = 0.2

    b[2] = 0.3

    weights = b.view(1, 2, 3)

    print("weights:", weights)

    y = F.conv1d(x, weights, padding=0)

    print("y:", y)


if __name__ == '__main__':
    tst_multi_channel()
