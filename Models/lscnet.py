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

    def __init__(self, indepth, outdepth, stages, kernel=16, levels=1):
        super(LSCNet, self).__init__()
        self.indepth = indepth
        self.outdepth = outdepth
        self.kernel = kernel
        self.levels = levels
        assert len(stages) == 1
        self.stages = stages[0]

        self.conv_left = nn.Sequential(
            nn.Conv2d(indepth, outdepth, (kernel, 1), stride=1, padding=0),
            nn.Conv2d(indepth, outdepth, (1, kernel), stride=1, padding=0)
        )
        self.conv_right = nn.Sequential(
            nn.Conv2d(indepth, outdepth, (1, kernel), stride=1, padding=0),
            nn.Conv2d(indepth, outdepth, (kernel, 1), stride=1, padding=0)
        )

    def forward(self, feature_maps):
        Cx = feature_maps[self.stages]
        # 左分支   # 右分支
        fmap_left = self.conv_left(Cx)
        fmap_right = self.conv_right(Cx)
        fmap = fmap_left + fmap_right
        return [fmap]

    def __repr__(self):
        return self.__class__


class NoneNet(nn.Module):
    def __init__(self, indepth, outdepth, stages):
        super(NoneNet, self).__init__()
        self.indepth = indepth
        self.outdepth = outdepth
        assert len(stages) == 1
        self.stages = stages[0]
        # self.none = nn.Sequential(
        #     nn.Conv2d(indepth, outdepth, (1, 1), stride=1, padding=0)
        # )

    def forward(self, feature_maps):
        return [feature_maps[self.stages]]
