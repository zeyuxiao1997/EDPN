#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   inferenceTools.py
@Time    :   2021/05/07 21:40:07
@Author  :   Zeyu Xiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   
'''
# here put the import lib

import os
from PIL import Image,ImageOps
import torch.utils.data as data
import torch
from torchvision.transforms import Compose, ToTensor
import os
import random
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import cv2
import numpy as np

def transform():
    return Compose([
        ToTensor(),
    ])

def get_image(img):
    img = Image.open(img).convert('RGB')
    return img


def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    #flow = rescale_flow(flow,0,1)
    return flow


def load_image_test_vimeo(LrRoot, GtRoot, videoID):
    LrDir = os.path.join(LrRoot, videoID)
    GtDir = os.path.join(GtRoot, videoID)
    inputs = [get_image(os.path.join(LrDir, 'im'+str(i+1)+'.png')) for i in range(7)]
    targets = [get_image(os.path.join(GtDir, 'im'+str(i+1)+'.png')) for i in range(7)]

    targets = [transform()(i) for i in targets]
    inputs = [transform()(j) for j in inputs]

    inputs = torch.cat((torch.unsqueeze(inputs[0], 0), torch.unsqueeze(inputs[1], 0),
                            torch.unsqueeze(inputs[2], 0), torch.unsqueeze(inputs[3], 0),
                            torch.unsqueeze(inputs[4], 0), torch.unsqueeze(inputs[5], 0),
                            torch.unsqueeze(inputs[6], 0)))

    targets = torch.cat((torch.unsqueeze(targets[0], 0), torch.unsqueeze(targets[1], 0),
                            torch.unsqueeze(targets[2], 0), torch.unsqueeze(targets[3], 0),
                            torch.unsqueeze(targets[4], 0), torch.unsqueeze(targets[5], 0),
                            torch.unsqueeze(targets[6], 0)))
    inputs = torch.unsqueeze(inputs,0)
    targets = torch.unsqueeze(targets,0)
    return inputs, targets


def load_image_test_vimeo_RBPN(LrRoot, GtRoot, videoID, idx):
    LrDir = os.path.join(LrRoot, videoID)
    GtDir = os.path.join(GtRoot, videoID)
    inputs = [get_image(os.path.join(LrDir, 'im'+str(i+1)+'.png')) for i in range(7)]
    targets = [get_image(os.path.join(GtDir, 'im'+str(i+1)+'.png')) for i in range(7)]

    if idx == 0:
        inputLR = inputs[0]
        target = targets[0]
        neighbors = [inputs[0], inputs[0], inputs[0], inputs[1],inputs[2],inputs[3]]
        flow = [get_flow(inputLR,j) for j in neighbors]
    
    if idx == 1:
        inputLR = inputs[1]
        target = targets[1]
        neighbors = [inputs[0], inputs[0], inputs[0], inputs[2],inputs[3],inputs[4]]
        flow = [get_flow(inputLR,j) for j in neighbors]

    if idx == 2:
        inputLR = inputs[2]
        target = targets[2]
        neighbors = [inputs[0], inputs[0], inputs[1], inputs[3],inputs[4],inputs[5]]
        flow = [get_flow(inputLR,j) for j in neighbors]

    if idx == 3:
        inputLR = inputs[3]
        target = targets[3]
        neighbors = [inputs[0], inputs[1], inputs[2], inputs[4],inputs[5],inputs[6]]
        flow = [get_flow(inputLR,j) for j in neighbors]
    
    if idx == 4:
        inputLR = inputs[4]
        target = targets[4]
        neighbors = [inputs[1], inputs[2], inputs[3], inputs[5],inputs[6],inputs[6]]
        flow = [get_flow(inputLR,j) for j in neighbors]
    
    if idx == 5:
        inputLR = inputs[5]
        target = targets[5]
        neighbors = [inputs[2], inputs[3], inputs[4], inputs[6],inputs[6],inputs[6]]
        flow = [get_flow(inputLR,j) for j in neighbors]

    if idx == 6:
        inputLR = inputs[6]
        target = targets[6]
        neighbors = [inputs[3], inputs[4], inputs[5], inputs[6],inputs[6],inputs[6]]
        flow = [get_flow(inputLR,j) for j in neighbors]

    return inputLR, target, neighbors, flow



def getLRGTpairs(seq_LR, seq_GT, index):
    LR1 = seq_LR[:, 0, :, :, :]
    LR2 = seq_LR[:, 1, :, :, :]
    LR3 = seq_LR[:, 2, :, :, :]
    LR4 = seq_LR[:, 3, :, :, :]
    LR5 = seq_LR[:, 4, :, :, :]
    LR6 = seq_LR[:, 5, :, :, :]
    LR7 = seq_LR[:, 6, :, :, :]
    GT1 = seq_GT[:, 0, :, :, :]
    GT2 = seq_GT[:, 1, :, :, :]
    GT3 = seq_GT[:, 2, :, :, :]
    GT4 = seq_GT[:, 3, :, :, :]
    GT5 = seq_GT[:, 4, :, :, :]
    GT6 = seq_GT[:, 5, :, :, :]
    GT7 = seq_GT[:, 6, :, :, :]
    if index==0:
        LR = torch.cat((torch.unsqueeze(LR1, 1),torch.unsqueeze(LR1, 1), torch.unsqueeze(LR1, 1),
                          torch.unsqueeze(LR1, 1),torch.unsqueeze(LR2, 1),torch.unsqueeze(LR3, 1),
                          torch.unsqueeze(LR4, 1)),1)
        GT = GT1
        # LR4T = torch.cat((torch.unsqueeze(LR1, 1),torch.unsqueeze(LR1, 1), torch.unsqueeze(LR1, 1),
        #                   torch.unsqueeze(LR1, 1),torch.unsqueeze(LR2, 1),torch.unsqueeze(LR3, 1),
        #                   torch.unsqueeze(LR4, 1)),1)
        # GT4T = GT1
    if index==1:
        LR = torch.cat((torch.unsqueeze(LR1, 1),torch.unsqueeze(LR1, 1), torch.unsqueeze(LR1, 1),
                          torch.unsqueeze(LR2, 1),torch.unsqueeze(LR3, 1),torch.unsqueeze(LR4, 1),
                          torch.unsqueeze(LR5, 1)),1)
        GT = GT2
        # LR4T = torch.cat((torch.unsqueeze(LR1, 1),torch.unsqueeze(LR1, 1), torch.unsqueeze(LR1, 1),
        #                   torch.unsqueeze(LR2, 1),torch.unsqueeze(LR3, 1),torch.unsqueeze(LR4, 1),
        #                   torch.unsqueeze(LR5, 1)),1)
        # GT4T = GT2
    if index==2:
        LR = torch.cat((torch.unsqueeze(LR1, 1),torch.unsqueeze(LR1, 1), torch.unsqueeze(LR2, 1),
                          torch.unsqueeze(LR3, 1),torch.unsqueeze(LR4, 1),torch.unsqueeze(LR5, 1),
                          torch.unsqueeze(LR6, 1)),1)
        GT = GT3
        # LR4T = torch.cat((torch.unsqueeze(LR1, 1),torch.unsqueeze(LR1, 1), torch.unsqueeze(LR2, 1),
        #                   torch.unsqueeze(LR3, 1),torch.unsqueeze(LR4, 1),torch.unsqueeze(LR5, 1),
        #                   torch.unsqueeze(LR6, 1)),1)
        # GT4T = GT3
    if index==3:
        LR = torch.cat((torch.unsqueeze(LR1, 1),torch.unsqueeze(LR2, 1), torch.unsqueeze(LR3, 1),
                          torch.unsqueeze(LR4, 1),torch.unsqueeze(LR5, 1),torch.unsqueeze(LR6, 1),
                          torch.unsqueeze(LR7, 1)),1)
        GT = GT4
        # LR4T = torch.cat((torch.unsqueeze(LR1, 1),torch.unsqueeze(LR2, 1), torch.unsqueeze(LR3, 1),
        #                   torch.unsqueeze(LR4, 1),torch.unsqueeze(LR5, 1),torch.unsqueeze(LR6, 1),
        #                   torch.unsqueeze(LR7, 1)),1)
        # GT4T = GT4
    if index==4:
        LR = torch.cat((torch.unsqueeze(LR2, 1),torch.unsqueeze(LR3, 1), torch.unsqueeze(LR4, 1),
                          torch.unsqueeze(LR5, 1),torch.unsqueeze(LR6, 1),torch.unsqueeze(LR7, 1),
                          torch.unsqueeze(LR7, 1)),1)
        GT = GT5
        # LR4T = torch.cat((torch.unsqueeze(LR2, 1),torch.unsqueeze(LR3, 1), torch.unsqueeze(LR4, 1),
        #                   torch.unsqueeze(LR5, 1),torch.unsqueeze(LR6, 1),torch.unsqueeze(LR7, 1),
        #                   torch.unsqueeze(LR7, 1)),1)
        # GT4T = GT5
    if index==5:
        LR = torch.cat((torch.unsqueeze(LR3, 1),torch.unsqueeze(LR4, 1), torch.unsqueeze(LR5, 1),
                          torch.unsqueeze(LR6, 1),torch.unsqueeze(LR7, 1),torch.unsqueeze(LR7, 1),
                          torch.unsqueeze(LR7, 1)),1)
        GT = GT6
        # LR4T = torch.cat((torch.unsqueeze(LR3, 1),torch.unsqueeze(LR4, 1), torch.unsqueeze(LR5, 1),
        #                   torch.unsqueeze(LR6, 1),torch.unsqueeze(LR7, 1),torch.unsqueeze(LR7, 1),
        #                   torch.unsqueeze(LR7, 1)),1)
        # GT4T = GT6
    if index==6:
        LR = torch.cat((torch.unsqueeze(LR4, 1),torch.unsqueeze(LR5, 1), torch.unsqueeze(LR6, 1),
                          torch.unsqueeze(LR7, 1),torch.unsqueeze(LR7, 1),torch.unsqueeze(LR7, 1),
                          torch.unsqueeze(LR7, 1)),1)
        GT = GT7
        # LR4T = torch.cat((torch.unsqueeze(LR4, 1),torch.unsqueeze(LR5, 1), torch.unsqueeze(LR6, 1),
        #                   torch.unsqueeze(LR7, 1),torch.unsqueeze(LR7, 1),torch.unsqueeze(LR7, 1),
        #                   torch.unsqueeze(LR7, 1)),1)
        # GT4T = GT7
    return LR, GT


def get_image(img):
    img = Image.open(img).convert('RGB')
    return img


def load_image_train2(group):
    images = [get_image(img) for img in group]
    inputs = images[0]
    target = images[1]
    return inputs, target

def quantize(img):
    return img.clip(0, 255).round().astype(np.uint8)