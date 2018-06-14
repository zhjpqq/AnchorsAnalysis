#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/4/6 14:39'
__author__ = 'ooo'

import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from Utils import utils
from torch.autograd import Variable
from torch.nn.modules.utils import _pair

from RoiTransforms.roi_align.modules.roi_align import RoIAlign, RoIAlignAvg, RoIAlignMax
from RoiTransforms.roi_crop.modules.roi_crop import RoICropFunction
from RoiTransforms.roi_pooling.modules.roi_pool import _RoIPooling, RoIPoolFunction
from Layers.rois_transform import pyramid_roi_align, parallel_roi_align


class ClassBoxNet(nn.Module):
    """
    fmap_stride: 原图到最后特征图的缩放比例
    """

    def __init__(self, indepth, pool_size, image_shape, fmap_stride, class_nums, level_nums):
        super(ClassBoxNet, self).__init__()
        self.indepth = indepth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.class_nums = class_nums
        self.fmap_stride = fmap_stride
        self.level_nums = level_nums

        self.conv1 = nn.Conv2d(self.indepth, 1024, kernel_size=pool_size, stride=1)  # 全连接
        self.bn1 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.classifier = nn.Linear(1024, class_nums)
        self.softmax = nn.Softmax(dim=1)

        self.bounding_box = nn.Linear(1024, class_nums * 4)

    def forward(self, x, rois):
        """
        levels=1, levels>1
        RoiAlign: -> [batch*N, channel, h', w']
        pyramid_roi_align: -> [batch*N, h', w', channel]
        :param
            x: [batch, channel, h, w], [x1, x2, x3, ...]
            rois: [batch, N, (y1, x1, y2, x2)], [rois1, rois2, ...]
        :return
            class_logits: [batch*N, class_nums, (logits)]
            class_probs:  [batch*N, class_nums, (probs)]
            bbox_deltas:  [batch*N, class_nums, (dy, dx, dw, dh)]
        """
        # todo ??? 返回Shape: [batch×num_rois, channels, pool_height, pool_width]

        if self.level_nums == 1 and len(x) == 1 and len(rois) == 1:
            rois = rois_expand(rois[0]).view(-1, 5)  # shape: [batch*N, (batch_index, y2, x2, y1, x1)]
            x = RoIAlignAvg(self.pool_size[0], self.pool_size[1], self.fmap_stride[0])(x[0], rois)

        elif self.level_nums > 1 and len(x) > 1 and len(rois) == 1:
            x = pyramid_roi_align(x, rois[0], self.pool_size, self.image_shape)

        elif self.level_nums > 1 and len(x) > 1and len(rois) > 1:
            # 并行计算, 多对多
            x = parallel_roi_align(x, rois, self.pool_size, self.image_shape)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x.view(-1, 1024)

        class_logits = self.classifier(x)
        class_probs = self.softmax(class_logits)

        bbox_deltas = self.bounding_box(x)
        bbox_deltas = bbox_deltas.view(bbox_deltas.size(0), -1, 4)

        return [class_logits, class_probs, bbox_deltas]

    def __repr__(self):
        return self.__class__.__name__


class MaskNet(nn.Module):
    def __init__(self, indepth, pool_size, image_shape, fmap_stride, class_nums, level_nums):
        super(MaskNet, self).__init__()
        self.indepth = indepth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.class_nums = class_nums
        self.fmap_stride = fmap_stride
        self.level_nums = level_nums

        self.padding = SamePad2d(kernel_size=3, stride=1)

        self.conv1 = nn.Conv2d(self.indepth, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, class_nums, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, rois):
        """
        levels=1, levels>1
        :param
            x: [batch, c, h, w],                 [x1, x2, ...]
            rois: [batch, N, (y1, x1, y2, x2)]   [rois1, rois2, ...]
        :return
            x: [batch*N, class_nums, h', w']
        """
        if self.level_nums == 1 and len(x) == 1 and len(rois) == 1:
            rois = rois_expand(rois[0]).view(-1, 5)
            x = RoIAlignAvg(self.pool_size[0], self.pool_size[1], self.fmap_stride[0])(x[0], rois)

        elif self.level_nums > 1 and len(x) > 1 and len(rois) == 1:
            x = pyramid_roi_align(x, rois[0], self.pool_size, self.image_shape)

        elif self.level_nums > 1 and len(x) > 1 and len(rois) > 1:
            # 并行计算
            x = parallel_roi_align(x, rois, self.pool_size, self.image_shape)

        x = self.conv1(self.padding(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(self.padding(x))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(self.padding(x))
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(self.padding(x))
        x = self.bn4(x)
        x = self.relu(x)
        x = self.deconv(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.sigmoid(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, input):
        in_width = input.size(2)
        in_height = input.size(3)
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


def rois_expand(rois):
    """
    :param rois: [batch, N, (y1, x1, y2, x2)] , tensor
    :return rois: [batch, N, (batch_index, y1, x1, y2, x2)] , tensor
    """
    batch, count = rois.shape[0:2]
    expand = Variable(torch.arange(0, batch), requires_grad=False).cuda()  # shape batch
    expand = expand.unsqueeze(-1).unsqueeze(-1).expand(batch, count, 1)  # shape batch x count x 1
    rois = torch.cat([expand, rois], dim=2)
    return rois
