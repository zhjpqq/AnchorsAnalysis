#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/6/7 7:54'
__author__ = 'ooo'

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from Utils import utils
import math
from RoiTransforms.roi_align2.crop_and_resize import CropAndResizeFunction


class RoiTransformLayer(nn.Module):
    def __init__(self):
        super(RoiTransformLayer, self).__init__()

    def forward(self, *inputs):
        pass
        return inputs


def pyramid_roi_align(feature_maps, rois, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.
       https://github.com/multimodallearning/pytorch-mask-rcnn/blob/master/model.py

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels]. # todo ??? 通道位置？
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    # Currently only supports batchsize 1
    # for i in range(len(inputs)):
    #     inputs[i] = inputs[i].squeeze(0)

    # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = rois

    # Feature Maps. List of feature maps from different level of the
    # feature pyramid. Each is [batch, height, width, channels]
    feature_maps = feature_maps

    # Assign each ROI to a level in the pyramid based on the ROI area.
    y1, x1, y2, x2 = boxes.chunk(4, dim=2)  # x,y shape: [batch, num_boxes, 1]
    h, w = y2 - y1, x2 - x1  # h,w shape: [batch, num_boxes, 1]
    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4
    image_area = Variable(torch.FloatTensor([float(image_shape[0] * image_shape[1])]), requires_grad=False).cuda()
    roi_level = 4 + log2(torch.sqrt(h * w) / (224.0 / torch.sqrt(image_area)))
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2, 5)  # shape: [batch, num_boxes, 1]
    roi_level = roi_level.squeeze(2)

    # 遍历各个batch
    all_pooled = []
    for b in range(roi_level.shape[0]):
        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = roi_level[b] == level  # ix, shape: [num_boxes]
            if not ix.any():
                continue
            ix = torch.nonzero(ix)[:, 0]  # ix, shape: [num_boxes]
            level_boxes = boxes[b][ix.data, :]  # level_boxes: [num_boxes, (y1, x1, y2, x2)]

            # Keep track of which box is mapped to which level, 索引跟踪
            box_to_level.append(ix.data)

            # Stop gradient propogation to ROI proposals
            level_boxes = level_boxes.detach()

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]    # todo ??? channels
            ind = b * Variable(torch.ones(level_boxes.size()[0]), requires_grad=False).int().cuda()  # todo ??? b
            # feature_maps[i] = feature_maps[i].unsqueeze(0)  # CropAndResizeFunction needs batch dimension
            pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(feature_maps[i], level_boxes, ind)
            pooled.append(pooled_features)

        # Pack pooled features into one tensor, 将各个level上裁剪出来的rois叠在一起.
        pooled = torch.cat(pooled, dim=0)  # shape [N, channel, pool_size, pool_size]

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes，
        # 将各个level上裁剪出的rois的索引叠在一起，原始索引 [2,3,4; 5,6,7,8; 0,1,9; ...]
        box_to_level = torch.cat(box_to_level, dim=0)

        # Rearrange pooled features to match the order of the original boxes
        # 只需要再对原始索引重新排序，就能将rois倒腾回原来的顺序。索引的索引，就回到原来的顺序.
        _, box_to_level = torch.sort(box_to_level)
        pooled = pooled[box_to_level, :, :]

        all_pooled.append(pooled)

    # 将所有batch的pooled叠在一起，构成 P：[batch*N, channel, pool_size, pool_size]
    # 在后续计算中，batch*N 相当于新的batch，其batch*N 个rois在计算过程上是相互独立的，计算结果互不影响。
    # 因此，只要 P.view(batch, N, pool_size, pool_zie)，就能复原计算结果.
    all_pooled = torch.cat(all_pooled, dim=0)

    return all_pooled


def parallel_roi_align(x, rois, pool_size, image_shape):
    # 多层特征图，每一层都有各自的rois, 因而需要并行处理，并行调用ROIAlign
    # x [P1, P2, P3, P4, P5]
    # rois [rois1, rois2, rois3, rois4, rois5]
    raise NotImplementedError


def log2(x):
    """Implementatin of Log2. Pytorch doesn't have a native implemenation."""
    ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
    if x.is_cuda:
        ln2 = ln2.cuda()
    return torch.log(x) / ln2
