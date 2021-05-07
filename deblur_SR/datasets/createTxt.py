#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   createTxt.py
@Time    :   2020/07/20 09:45:56
@Author  :   ZeyuXiao 
@Version :   1.0
@Contact :   zeyuxiao1997@163.com
@License :   (C)Copyright 2018-2019
@Desc    :   create .txt for dataloader
'''
# here put the import lib

import os
import argparse


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def main():
    # 7 frames
    assert os.path.exists(inputdir), 'Input dir not found'
    assert os.path.exists(targetdir), 'target dir not found'
    mkdir(outputdir)
    vids = sorted(os.listdir(inputdir))
    
    for vid in vids:
        for idx in range(0, 100):
            groups = ''
            
            groups += os.path.join(inputdir, vid, '{:08d}'.format(idx) + ext) + '|'
            groups += os.path.join(targetdir, vid, '{:08d}'.format(idx+2) + ext)
            
            with open(os.path.join(outputdir, 'groups.txt'), 'a') as f:
                f.write(groups + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/gdata/xiaozy/NTIRE2021/deblur_SR/train_blur_bicubic', metavar='PATH', help='root dir to save low-resolution images (degraded images)')
    parser.add_argument('--target', type=str, default='/gdata/xiaozy/NTIRE2021/GroundTruth/train_sharp', metavar='PATH', help='root dir to save high-resolution images (ground truth)')
    parser.add_argument('--output', type=str, default='/gdata/xiaozy/NTIRE2021/deblur_SR', metavar='PATH', help='output dir to save group txt files')
    parser.add_argument('--ext', type=str, default='.png', help='Extension of files')
    args = parser.parse_args()

    inputdir = args.input
    targetdir = args.target
    outputdir = args.output
    ext = args.ext

    main()