#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   inference.py
@Time    :   2021/05/07 21:38:41
@Author  :   Zeyu Xiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   
'''
# here put the import lib

import argparse
import torch
import os
import numpy as np
import inferenceTools
import time
from PIL import Image, ImageOps
from models import EDPN
from utils.myutils import *
from torchvision.transforms import Compose,ToPILImage,ToTensor
import skimage.color as sc

parser = argparse.ArgumentParser(description='InferenceDeblockingSR')
# change the save folder name
parser.add_argument('--ModelName', type=str, default='VIDAR_Team', help='saveFolderName')
parser.add_argument('--scale', type=int, default=4, help="SR scale=4")
parser.add_argument('--in_channels', type=int, default=3, help='input channels')
parser.add_argument('--out_channels', type=int, default=3, help='output channels')
# change the saveRoot name
parser.add_argument('--saveRoot', type=str, default='/gdata/xiaozy/inferenceResultsDeblurSR', help='saveroot')
# change the degraded images path name
parser.add_argument('--lrRoot', type=str, default='/gdata/xiaozy/NTIRE2021/deblur_SR/test_blur_bicubic', help='degraded image path')

################################################################################
# EDVR parameters
################################################################################
parser.add_argument('--nf', type=int, default=64, help='nf')
parser.add_argument('--embed_ch', type=int, default=64, help='nf')
parser.add_argument('--nframes', type=int, default=7, help='total number of duplicated frames used for input')
parser.add_argument('--groups', type=int, default=8, help='number of groups')
parser.add_argument('--front_RBs', type=int, default=5, help='number of feature extractor backbones')
parser.add_argument('--back_RBs', type=int, default=10, help='number of reconstruction backbones')
parser.add_argument('--w_TSA', type=bool, default=True, help='input batch size')

# change the .pth path
parser.add_argument('--SRcheckpoint', type=str, default='xxxxx.pth', help='the path of .pth file(s)')

opt = parser.parse_args()

# print(opt)


def transform():
    return Compose([
        ToTensor(),
    ])


def inference():
    opt.n_SPA_blocks = 2
    opt.nframes = 5
    opt.groups = 8
    opt.front_RBs = 18
    opt.back_RBs = 120

    lrRoot = opt.lrRoot
    saveRoot = opt.saveRoot
    ModelName = opt.ModelName
    saveDir = os.path.join(saveRoot,ModelName)
    print(saveDir)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    

    # loading models #
    modelDeSR = EDPN.EDPN(opt)
    modelDeSR.eval()
    modelDeSR.cuda()

    if opt.SRcheckpoint != '':
        print('=========> loading trained Refine model for refining')
        map_location = lambda storage, loc: storage
        checkpoint = torch.load(opt.SRcheckpoint, map_location=map_location)
        optimizer_state = checkpoint["model"]
        modelDeSR.load_state_dict(optimizer_state)
        print('======> load trained Refine model for refining')



    ###############################################################################################
    # data loading part
    ###############################################################################################

    avgtime = 0


    for i in range(30):
        for j in range(9,100,10):
            lrfilepath = os.path.join(lrRoot,str(i).zfill(3),str(j).zfill(8)+'.png')
            # print(lrfilepath)

            saveDir = os.path.join(saveDir)
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
        
            inputs = Image.open(lrfilepath).convert('RGB')

            inputs = transform()(inputs)

            inputs = torch.unsqueeze(inputs,0)
            inputs = torch.cat((torch.unsqueeze(inputs, 1), torch.unsqueeze(inputs, 1), torch.unsqueeze(inputs, 1), torch.unsqueeze(inputs, 1), torch.unsqueeze(inputs, 1)),1)
            
            inputs = inputs.cuda()


            with torch.no_grad():
                out = modelDeSR(inputs)  
            out = torch.clamp(out,0.0,1.0).cpu()
            torch.cuda.empty_cache()

            R = ToPILImage()(out[0]).convert('RGB') 

            R.save(os.path.join(saveDir, str(i).zfill(3) + '_'+str(j).zfill(8) + '.png'))
            
            print(os.path.join(saveDir, str(i).zfill(3) + '_'+str(j).zfill(8) + '.png'),'done')

if __name__ == "__main__":
    inference()
