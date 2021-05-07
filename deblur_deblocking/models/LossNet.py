#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   LossNet.py
@Time    :   2021/05/07 21:42:57
@Author  :   Zeyu Xiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   
'''
# here put the import lib

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchvision import models

from torchvision import models
import torchvision

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss

if __name__ == "__main__":
    images = Variable(torch.ones(1, 3, 128, 128)).cuda()
    vgg = Vgg19()
    vgg.cuda()
    print("do forward...")
    outputs = vgg(images)
    print (outputs.size())   # (10, 100)
    print(torch.max(outputs))
