#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train1.py
@Time    :   2021/05/07 21:40:22
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

import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
from models import Losses
from models import EDPN
from utils.myutils import *
import torch.backends.cudnn as cudnn
from options import *
from datasets.REDS4DeblurDeblocking import *
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
import random
import numpy as np
from tensorboardX import *
import torchvision.utils as visionutils
import math
# from thop import profile


def train():
    opt.n_SPA_blocks = 3
    opt.nframes = 5
    opt.groups = 8
    opt.front_RBs = 18
    opt.back_RBs = 120
    opt.train_batch_size = 4
    opt.num_workers = 8

    print(opt)
    Best = 0
    transform = transforms.Compose([transforms.ToTensor()])
    opt.manualSeed = random.randint(1, 10000)
    opt.saveDir = os.path.join(opt.exp, opt.ModelName)
    create_exp_dir(opt.saveDir)
    device = torch.device("cuda:7")

    train_data = DatasetFromFolder(opt)
    train_dataloader = DataLoader(train_data,
                        batch_size=opt.train_batch_size,
                        shuffle=True,
                        num_workers=opt.num_workers,
                        drop_last=True)
    print('length of train_dataloader: ',len(train_dataloader)) # 6000
    last_epoch = 0

    ## initialize loss writer and logger
    ##############################################################
    loss_dir = os.path.join(opt.saveDir, 'loss')
    loss_writer = SummaryWriter(loss_dir)
    print("loss dir", loss_dir)
    trainLogger = open('%s/train.log' % opt.saveDir, 'w')
    ##############################################################

    model = EDPN.EDPN(opt)
    model.train()
    model.cuda()

    criterionCharb = Losses.CharbonnierLoss()
    criterionCharb.cuda()


    lr = opt.lr
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr,
        betas=(opt.beta1, opt.beta2)
    )

    iteration = 0

    for epoch in range(opt.max_epoch):
        if epoch < last_epoch:
            continue
        for _, batch in enumerate(train_dataloader, 0):
            iteration += 1 

            inputs, target = batch
            inputs, target = inputs.cuda(), target.cuda()
            inputs = torch.cat((torch.unsqueeze(inputs, 1), torch.unsqueeze(inputs, 1), torch.unsqueeze(inputs, 1), torch.unsqueeze(inputs, 1), torch.unsqueeze(inputs, 1)),1)
            out = model(inputs)
            
            optimizer.zero_grad()
            
            CharbLoss1 = criterionCharb(out, target)
            AllLoss = CharbLoss1
            AllLoss.backward()
            optimizer.step()

            prediction = torch.clamp(out,0.0,1.0)

            if iteration%2 == 0:
                PPsnr = compute_psnr(tensor2np(prediction[0,:,:,:]),tensor2np(target[0,:,:,:]))
                if PPsnr==float('inf'):
                    PPsnr=99
                AllPSNR += PPsnr
                print('[%d/%d][%d] AllLoss:%.10f|CharbLoss:%.10f|PSNR:%.6f'
                    % (epoch, opt.max_epoch, iteration,
                    AllLoss.item(),CharbLoss1.item(), PPsnr))
                trainLogger.write(('[%d/%d][%d] AllLoss:%.10f|CharbLoss:%.10f|PSNR:%.6f'
                    % (epoch, opt.max_epoch, iteration,
                    AllLoss.item(),CharbLoss1.item(), PPsnr))+'\n')

                loss_writer.add_scalar('CharbLoss', CharbLoss1.item(), iteration)
                loss_writer.add_scalar('PSNR', PPsnr, iteration)
                trainLogger.flush()

            if iteration%5000 == 0:
                loss_writer.add_image('Prediction', prediction[0,:,:,:], iteration) # x.size= (3, 266, 530) (C*H*W)
                loss_writer.add_image('target', target[0,:,:,:], iteration)

                
            if iteration % opt.saveStep == 0:
                is_best = AllPSNR > Best
                Best = max(AllPSNR, Best)
                if is_best or iteration%(opt.saveStep*5)==0:
                    prefix = opt.saveDir+'/DeblurSR_iter{}'.format(iteration)+'+PSNR'+str(Best)
                    file_name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
                    checkpoint = {
                        'epoch': epoch,
                        'iteration': iteration,
                        "optimizer": optimizer.state_dict(),
                        "model": model.state_dict(),
                        "lr": lr
                    }
                torch.save(checkpoint, file_name)
                print('model saved to ==>'+file_name)
                AllPSNR = 0

            if (iteration + 1) % opt.decay_step == 0:
                lr = lr * opt.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

    trainLogger.close()



if __name__ == "__main__":
    train()
