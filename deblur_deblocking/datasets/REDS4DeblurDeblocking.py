#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   REDS4DeblurDeblocking.py
@Time    :   2021/05/07 21:59:02
@Author  :   Zeyu Xiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   
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


def get_patch(img_nn, img_tar, patch_size, scale=1, ix=-1, iy=-1):
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


    img_tar = img_tar.crop((iy, ix, iy + ip, ix + ip))  # [:, ty:ty + tp, tx:tx + tp]
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
        self.transform = transform
        self.data_augmentation = opt.augmentation
        self.patch_size = opt.patch_size
        self.hflip = opt.hflip
        self.rot = opt.rot

    def __getitem__(self, index):

        inputs, target = load_image_train2(self.image_filenames[index])

        if self.patch_size != 0:
            inputs, target = get_patch(inputs, target, self.patch_size)

        if self.data_augmentation:
            inputs, target = augment(inputs, target, self.hflip, self.rot)

        if self.transform:
            target = self.transform(target)
            inputs = self.transform(inputs)


        return inputs, target

    def __len__(self):
        return len(self.image_filenames)




if __name__ == '__main__':
    output = 'visualize'
    if not os.path.exists(output):
        os.mkdir(output)
    dataset = DatasetFromFolder(4, True, 'dataset/groups.txt', 64, True, True, True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=False)
    for i, (inputs, target) in enumerate(dataloader):
        if i > 10:
            break
        if not os.path.exists(os.path.join(output, 'group{}'.format(i))):
            os.mkdir(os.path.join(output, 'group{}'.format(i)))
        input0, input1, input2, input3, input4 = inputs[0][0], inputs[0][1], inputs[0][2], inputs[0][3], inputs[0][4]
        vutils.save_image(input0, os.path.join(output, 'group{}'.format(i), 'input0.png'))
        vutils.save_image(input1, os.path.join(output, 'group{}'.format(i), 'input1.png'))
        vutils.save_image(input2, os.path.join(output, 'group{}'.format(i), 'input2.png'))
        vutils.save_image(input3, os.path.join(output, 'group{}'.format(i), 'input3.png'))
        vutils.save_image(input4, os.path.join(output, 'group{}'.format(i), 'input4.png'))
        vutils.save_image(target, os.path.join(output, 'group{}'.format(i), 'target.png'))