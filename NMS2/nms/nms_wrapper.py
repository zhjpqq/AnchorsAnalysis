# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# https://github.com/zhjpqq/pytorch-mask-rcnn/tree/master/nms

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .pth_nms import pth_nms


def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations.
    Accept dets as tensor"""
    return pth_nms(dets, thresh)
