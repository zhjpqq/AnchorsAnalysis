#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/4/3 21:43'
__author__ = 'ooo'

import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class LSCNet(nn.Module):
    """
    大尺度卷积网络
    """
    def __init__(self, indepth, outdepth=96, kernel=16, levels=1):
        super(LSCNet, self).__init__()
        self.indepth = indepth
        self.outdepth = outdepth
        self.kernel = kernel
        self.levels = levels

        self.conv_left = nn.Sequential(
            nn.Conv2d(indepth, outdepth, (kernel, 1), stride=1, padding=0),
            nn.Conv2d(indepth, outdepth, (1, kernel), stride=1, padding=0)
        )
        self.conv_right = nn.Sequential(
            nn.Conv2d(indepth, outdepth, (1, kernel), stride=1, padding=0),
            nn.Conv2d(indepth, outdepth, (kernel, 1), stride=1, padding=0)
        )

    def forward(self, feature_maps):
        C1, C2, C3, C4, C5 = feature_maps
        # 左分支   # 右分支
        fmap_left = self.conv_left(C5)
        fmap_right = self.conv_right(C5)
        fmap = [fmap_left + fmap_right]
        return [fmap]

    def __repr__(self):
        return self.__class__
