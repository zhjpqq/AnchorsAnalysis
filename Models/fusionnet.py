#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/6/5 21:29'
__author__ = 'ooo'

from Models.fpnet import FPNet
from Models.lscnet import LSCNet, NoneNet


def fusionnet(method, levels, stages, indepth, outdepth, strides, shapes):
    """
    :param method:
    :param levels:
    :param indepth:
    :param outdepth:
    :param strides:
    :param shapes:
    :param stages: which to choose in [C1, C2, C3, C4, C5, C6]
    :return:
    """
    if not isinstance(stages, (list, tuple)):
        stages = [stages]
    stages = [int(s[1]) - 1 for s in stages]
    if method == 'fpn':
        # [P2,P3,P4,P5,P6], P6 is upsampled form P5
        assert len(stages) == levels, '输入特征级数与指定的FUSION_LEVELS不匹配！'
        net = FPNet(indepth=indepth, outdepth=outdepth, stages=stages)

    elif method == 'lsc':
        net = LSCNet(indepth=indepth, outdepth=outdepth, stages=stages, kernel=16)

    elif method == 'none':
        net = NoneNet(indepth=indepth, outdepth=outdepth, stages=stages)

    else:
        raise ValueError('未知的参数设定！Unknown feature FUSION_METHOD！%s' % method)

    return net
