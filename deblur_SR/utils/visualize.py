#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   visualize.py
@Time    :   2021/05/07 21:06:37
@Author  :   Zeyu Xiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   
'''
# here put the import lib

import visdom
import time
import numpy as np

class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X= np.array([x]),
                      win=(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        self.vis.images(img_.cpu().numpy(),
                        win=(name),
                        opts=dict(title=name),
                        **kwargs)

    def log(self, info, win='log_text'):
        self.log_text += ('[{time}] {info} <br/>'.format(time=time.strftime('%m%d_%H:%M:%S'),
                                                         info=info))
        self.vis.text(self.log_text, win=win)

    def __getattr__(self, function_name):
        return getattr(self.vis, function_name)
