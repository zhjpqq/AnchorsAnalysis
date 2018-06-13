#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/6/5 21:29'
__author__ = 'ooo'

from Models.fpnet import FPNet
from Models.lscnet import LSCNet


def fusionnet(method, levels, indepth, outdepth, strides, shapes):
    if method == 'fpn':
        assert len(indepth) == levels-1, '输入特征级数与指定的FUSION_LEVELS不匹配！'
        net = FPNet(indepth=indepth, outdepth=outdepth)
    elif method == 'lsc':
        net = LSCNet(indepth=indepth, outdepth=outdepth, kernel=16)
    elif method == 'none':
        net = None
    else:
        raise ValueError('未知的参数设定！Unknown feature FUSION_METHOD！%s' % method)
    return net

