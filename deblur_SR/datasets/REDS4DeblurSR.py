#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   REDS4DeblurSR.py
@Time    :   2021/05/07 19:53:26
@Author  :   Zeyu Xiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   dataloader part
'''
# here put the import lib

import torch.utils.data as data
import torch
from torchvision.transforms import Compose, ToTensor
import os
import random
from PIL import Image,ImageOps
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import cv2
import numpy as np



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def get_patch(img_nn, img_tar, patch_size, scale=4, ix=-1, iy=-1):
    (ih, iw) = img_nn.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)


    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]
    img_nn = img_nn.crop((iy, ix, iy + ip, ix + ip))  # [:, iy:iy + ip, ix:ix + ip]

    return  img_nn, img_tar


def augment(img_nn, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_tar = ImageOps.flip(img_tar)
        img_nn = ImageOps.flip(img_nn)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_tar = ImageOps.mirror(img_tar)
            img_nn = ImageOps.mirror(img_nn)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_tar = img_tar.rotate(180)
            img_nn = img_nn.rotate(180)
            info_aug['trans'] = True

    return  img_nn, img_tar

def get_image(img):
    img = Image.open(img).convert('RGB')
    return img


def load_image_train2(group):
    images = [get_image(img) for img in group]
    inputs = images[0]
    target = images[1]
    return inputs, target


def transform():
    return Compose([
        ToTensor(),
    ])


class DatasetFromFolder(data.Dataset):
    """
    For test dataset, specify
    `group_file` parameter to target TXT file
    data_augmentation = None
    black_edge_crop = None
    flip = None
    rot = None
    """
    def __init__(self, opt, transform=transform()):
        super(DatasetFromFolder, self).__init__()
        groups = [line.rstrip() for line in open(os.path.join(opt.group_file))]
        # assert groups[0].startswith('/'), 'Paths from file_list must be absolute paths!'
        self.image_filenames = [group.split('|') for group in groups]
        self.upscale_factor = opt.upscale_factor
        self.transform = transform
        self.data_augmentation = opt.augmentation
        self.patch_size = opt.patch_size
        self.hflip = opt.hflip
        self.rot = opt.rot

    def __getitem__(self, index):

        inputs, target = load_image_train2(self.image_filenames[index])

        if self.patch_size != 0:
            inputs, target = get_patch(inputs, target, self.patch_size, self.upscale_factor)

        if self.data_augmentation:
            inputs, target = augment(inputs, target, self.hflip, self.rot)

        if self.transform:
            target = self.transform(target)
            inputs = self.transform(inputs)


        return inputs, target

    def __len__(self):
        return len(self.image_filenames)

