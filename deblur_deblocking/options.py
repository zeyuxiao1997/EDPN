#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   options.py
@Time    :   2021/05/07 22:03:06
@Author  :   Zeyu Xiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   
'''
# here put the import lib

import argparse
import os

parser = argparse.ArgumentParser(description='Temporal-Spatial-SR')

# general settings
parser.add_argument('--ModelName', type=str, default='EDPN', help='prefix of different dataset')
parser.add_argument('--exp', type=str, default='/output', help='prefix of different dataset')
parser.add_argument('--lr', type=float, default=1e-4, help='input batch size')
parser.add_argument('--lr_decay', type=float, default=0.5, help='input batch size')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for ADAM')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')
parser.add_argument('--loss_alpha', type=float, default=0.5, help='beta2 for ADAM')
parser.add_argument('--model_path', type=str, default='', help='number of epochs to train for')

parser.add_argument('--BreakCheckpoint', type=str, default='', help='number of epochs to train for')



# dataloader parameters
parser.add_argument('--embed_ch', type=int, default=64, help='file for labels')
parser.add_argument('--nf', type=int, default=160, help='file for labels')
parser.add_argument('--upscale_factor', type=int, default=4, help="optimizer [Options: SGD, ADAM]")
parser.add_argument('--group_file', type=str, default='/data/zeyuxiao1997/Image_deblur_xzy/groups.txt', help='file for labels')
parser.add_argument('--augmentation', type=bool, default=True, help='prefix of different dataset')
parser.add_argument('--patch_size', type=int, default=64, help='file for labels')
parser.add_argument('--hflip', type=bool, default=True, help='prefix of different dataset')
parser.add_argument('--rot', type=bool, default=True, help='prefix of different dataset')


# training parameters
parser.add_argument('--train_batch_size', type=int, default=4, help='file for labels')
parser.add_argument('--num_workers', type=int, default=0, help='file for val labels')
parser.add_argument('--max_epoch', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--decay_step', type=int, default=1, help='input batch size')
parser.add_argument('--saveStep', type=int, default=1, help='input batch size')

# EDVR parameters




opt = parser.parse_args()

def create_exp_dir(exp):
    try:
        os.makedirs(exp)
        print('Creating exp dir: %s' % exp)
    except OSError:
        pass
    return True
