#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/4/3 21:43'
__author__ = 'ooo'

import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class SimpleNet(nn.Module):

    def __init__(self, indepth, outdepth, kernel=3, levels=1, keep_same=True):
        super(SimpleNet, self).__init__()
        self.indepth = indepth
        self.outdepth = outdepth
        self.kernel = kernel
        self.keep_same = keep_same
        self.levels = levels

        self.conv = nn.Conv2d(indepth, outdepth, kernel, stride=1)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        # x = self.maxpool(x)
        return x

    def __repr__(self):
        return self.__class__