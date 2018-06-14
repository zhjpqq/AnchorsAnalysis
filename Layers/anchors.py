#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/6/5 22:41'
__author__ = 'ooo'

import numpy as np
import torch
from torch.autograd import Variable
from torch import nn


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, anchor_stride, feature_shape, feature_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    feature_shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    # 相对原图间隔为 anchor_stride*feature_stride
    shifts_y = np.arange(0, feature_shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, feature_shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)

    # 归一化，并裁剪，0~1
    image_shape = np.array([feature_shape[0], feature_shape[1], feature_shape[0], feature_shape[1]])*feature_stride
    boxes = np.clip(boxes/image_shape, 0, 1)

    return boxes


def generate_pyramid_anchors(scales, ratios, stride, counts, levels, zero_area, feature_shapes, feature_strides):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    parallel = False

    if levels == 1 and len(feature_shapes) == 1:
        anchors = generate_anchors(scales, ratios, stride, feature_shapes[0], feature_strides[0])

    elif levels > 1 and len(feature_shapes) > 1:
        # 将各层特征图上生成的anchors汇总在一起输出
        anchors = []
        for i in range(len(scales)):
            anchors.append(generate_anchors(scales[i], ratios, stride, feature_shapes[i], feature_strides[i]))
        anchors = np.concatenate(anchors, axis=0)

    elif levels > 1 and len(feature_shapes) > 1 and parallel:
        # 将各层特征图上生成的anchors独立输出
        anchors = []
        for i in range(len(scales)):
            level_anchors = generate_anchors(scales[i], ratios, stride, feature_shapes[i], feature_strides[i])
            level_anchors = Variable(torch.from_numpy(level_anchors).float(), requires_grad=False).cuda()
            anchors.append(level_anchors)
        return anchors

    else:
        raise ValueError('Error Pyramid Levels... %s' % levels)

    anchors = Variable(torch.from_numpy(anchors).float(), requires_grad=False).cuda()
    return [anchors]


class PyramidAnchorLayer(nn.Module):

    def __init__(self, scales, ratios, stride, counts, levels, zero_area, feature_shapes, feature_strides):
        super(PyramidAnchorLayer, self).__init__()
        self.scales = scales
        self.ratios = ratios
        self.stride = stride
        self.counts = counts
        self.levels = levels
        self.zero_area = zero_area
        self.feature_shapes = feature_shapes
        self.feature_strides = feature_strides

    def forward(self, *input):
        return generate_pyramid_anchors(self.scales, self.ratios, self.stride,
                                        self.counts, self.levels, self.zero_area,
                                        self.feature_shapes, self.feature_strides)

