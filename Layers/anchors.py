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

# Centers: generated respect to image_shape or fmap_shape*fmap_stride
# Scales/Ratios: set respect to image_shape
# so Anchors (y, x, h, w)/image_shape: Normalized in image_shape.
# then Anchors (y', x', h', w').clip(0, 1): Normalize in 0~1.
#
# but Small Objects will can't be found in Small Fmaps after pooling. So user level mapping in FPN.

class HotAnchorLayer(nn.Module):
    """
    scales setted relative to image_shape, anchors normalized in image_shape
    """

    def __init__(self, scales, ratios, stride, counts, levels, method, zero_area, image_shape, feature_shapes,
                 feature_strides):
        super(HotAnchorLayer, self).__init__()
        self.scales = scales
        self.ratios = ratios
        self.stride = stride  # anchor_stride
        self.counts = counts
        self.levels = levels
        self.heat_method = method
        self.zero_area = zero_area
        self.image_shape = image_shape
        self.feature_shapes = feature_shapes
        self.feature_strides = feature_strides

    def forward(self, feature_maps):
        """
        input:
        - feature_maps: [fmap1, fmap2, ...], 尺寸按大到小排列
          fmap1: [batch, c, h, w]，Variable

        output:
        - all_anchors: [anchors1, anchors2, ...], Variable
          anchors1: [batch, N, (y1, x1, y2, x2)] , array
        """
        # 热度统计方法
        if self.heat_method == 'accumulate':
            heat_centers = self.accumulate_heat_map
        elif self.heat_method == 'separable':
            heat_centers = self.seperable_heat_map
        elif self.heat_method == 'window':
            heat_centers = self.window_heat_map
        else:
            raise ValueError("未知的heat_method！")

        all_anchors, parallel = [], False

        # Note：单层/多层情况下，每个点上的scale不同，anchors中的数量N不相同！
        # all_anchors : counts*len(scales)*len(ratios)*level_nums
        # 1000*3*3*1  vs  1000*1*3*5

        # 单层特征图
        if self.levels == 1 and len(feature_maps) == 1:
            centers = heat_centers(feature_maps[0], self.counts, self.image_shape)
            anchors = self.boxes_generate(centers, self.scales, self.ratios, self.image_shape,
                                          feature_maps[0].shape[2:], self.zero_area)
            all_anchors.append(anchors)

        # 金字塔特征图, 综合输出
        elif self.levels > 1 and len(feature_maps) > 1 and not parallel:
            assert self.levels == len(self.scales), '特征层levels与scales不相同！'
            assert len(feature_maps) == len(self.scales), '特征图fmaps与scales层级不相同！'
            xanchors = []
            for fmap, scale in zip(feature_maps, self.scales):
                # assert scale < fmap.shape[2], '特征图尺寸不能比scale尺寸小'
                centers = heat_centers(fmap, self.counts, self.image_shape)
                anchors = self.boxes_generate(centers, scale, self.ratios, self.image_shape, fmap.shape[2:],
                                              self.zero_area)
                xanchors.append(anchors)
                anchors = None
            all_anchors.append(torch.cat(xanchors, dim=1))

        # 金字塔特征图，并行输出
        elif self.levels > 1 and len(feature_maps) > 1 and parallel:
            assert self.levels == len(self.scales), '特征层levels与scales不相同！'
            assert len(feature_maps) == len(self.scales), '特征图fmaps与scales层级不相同！'
            for fmap, scale in zip(feature_maps, self.scales):
                # assert scale < fmap.shape[2], '特征图尺寸不能比scale尺寸小'
                centers = heat_centers(fmap, self.counts, self.image_shape)
                anchors = self.boxes_generate(centers, scale, self.ratios, self.image_shape, fmap.shape[2:],
                                              self.zero_area)
                all_anchors.append(anchors)

        else:
            raise ValueError('错误的特征融合层数！%s')

        return all_anchors

    @staticmethod
    def accumulate_heat_map(x, counts, image_shape):
        """方案3
        累积最大热度：将所有通道独立去均值，独立求绝对值，再映射到01？，再累积相加，然后比大小.
        返回的centers，必须是以原图尺寸为参照！
        :param  x: [batch, channels, height, weight]，float32
        :param  shape: (h, w, c) 原图尺寸
        :return: centers: [batch, N, (y, x)] tensor, Normalized in image_shape.
        """
        assert x.size(2) * x.size(3) >= counts, 'need anchors counts >= pixels of fmap :%s & %s'
        strides = torch.FloatTensor([image_shape[0] / x.size(2), image_shape[1] / x.size(3)]).cuda()
        centers = x.data.new(x.size(0), counts, 2).zero_()
        for b in range(x.size(0)):
            fmap = x[b, :, :, :]
            fmap = torch.sum(torch.abs(fmap - torch.mean(torch.mean(fmap, 1, True), 2, True)), 0)
            # fmap = torch.sum(torch.abs(fmap), dim=0)
            kval, _ = torch.topk(fmap.view(-1), k=counts)
            points = np.where((fmap >= kval[-1]).data)  # maybe > counts
            points_y = torch.FloatTensor(points[0][0:counts]).contiguous().cuda()
            points_x = torch.FloatTensor(points[1][0:counts]).contiguous().cuda()
            points = torch.cat([points_y.view(-1, 1), points_x.view(-1, 1)], dim=1)
            centers[b, :, :] = points / strides
        return centers

    @staticmethod
    def seperable_heat_map(x, counts, shape):
        """ 方案1
        独立最大热度：各特征通道独立去均值，再独立求绝对值，再独立比大小，再全通道汇总
        """
        centers = []
        return centers

    @staticmethod
    def window_heat_map(x, counts, shape):
        '''
        pool option
        '''
        centers = []
        return centers

    @staticmethod
    def boxes_generate(centers, scales, ratios, image_shape, fmap_shape, zero_area):
        """
        在每个点上生成一组盒子，具有不同的尺度和形状，返回所有盒子角点坐标.
        scales, centers 都以原图尺寸为参照，因此生成的盒子最后/image_shape, 即可归一化！

        scales = array, [16, 32, 64, 128, 256, ...]，levels>1时，每层分配1个scale
        ratios = array, [0.5, 1, 2]
        centers = tensor, [batch, N, (y, x)]
        all_boxes = tensor, [batch, N*len(scales)*len(ratios), (y1, x1, y2, x2)]
        """
        if not isinstance(scales, (list, tuple)):
            scales = [scales]

        centers = centers.cpu().numpy()
        batches, counts = centers.shape[0:2]
        counts = counts * len(scales) * len(ratios)
        all_boxes = np.zeros(shape=(batches, counts, 4), dtype='float32')

        # 第一步：获取w/h组合
        # 获取所有scale和ratios的组合
        scales, ratios = np.meshgrid(scales, ratios)
        scales, ratios = scales.flatten(), ratios.flatten()  # shape N
        # 按ratios调整scales尺度
        heights = (scales / np.sqrt(ratios)).reshape(-1, 1)  # shape N
        widths = (scales * np.sqrt(ratios)).reshape(-1, 1)  # shape N

        for b in range(batches):
            # 第二步：获取y/x  shape: [N2, (y, x)]
            # box_centers = centers[b, :, :]
            # ctr_y, ctr_x = centers[b, :, 0], centers[b, :, 1]

            box_heights, box_centers_y = np.meshgrid(heights, centers[b, :, 0])  # shape NxK, NxK
            box_widths, box_centers_x = np.meshgrid(widths, centers[b, :, 1])

            # Reshape to get a list of (y, x) and a list of (h, w)
            box_centers = np.stack([box_centers_y, box_centers_x], axis=2)  # shape NxKx2
            box_sizes = np.stack([box_heights, box_widths], axis=2)  # shape NxKx2
            box_centers = np.reshape(box_centers, (-1, 2))
            box_sizes = np.reshape(box_sizes, (-1, 2))

            # 第四步：中心坐标+长宽 变换为 对角顶点坐标
            # Convert to corner coordinates (y1, x1, y2, x2)
            # np.concatenate([[y1, x1],[y2, x2]], axis=1) → [y1, x1, y2, x2]
            all_boxes[b, :, :] = np.concatenate([box_centers - 0.5 * box_sizes,
                                                 box_centers + 0.5 * box_sizes], axis=1)

        # filter 0 area boxes, object must have 4*4 pixels on original image
        # note : each batch has different counts of zero-area boxes

        # 归一化  [batch, counts, (y1, x1, y2, x2)]
        all_boxes /= np.array([image_shape[0], image_shape[1], image_shape[0], image_shape[1]]).astype(np.float32)

        # crop tp [0~1] within image shape
        all_boxes = torch.from_numpy(all_boxes).cuda().float().clamp(0, 1)

        # take out zero_area boxes
        # todo ??? : how to filter zero_area

        all_boxes = Variable(all_boxes, requires_grad=False)
        return all_boxes


class GeneralAnchorLayer(nn.Module):
    """
    scales setted relative to image_shape, anchors normalized in image_shape
    """

    def __init__(self, scales, ratios, stride, counts, levels, zero_area, image_shape, feature_shapes, feature_strides):
        super(GeneralAnchorLayer, self).__init__()
        self.scales = scales
        self.ratios = ratios
        self.stride = stride
        self.counts = counts
        self.levels = levels
        self.zero_area = zero_area
        self.image_shape = image_shape
        self.feature_shapes = feature_shapes
        self.feature_strides = feature_strides

    def forward(self, *input):
        """
        : input: fmaps in list. not used.
        : return: [A1, A2, A3, ...]
            anchors: [N, (y1, x1, y2, x2)], no batch specific!!!
        """
        return self.generate_pyramid_anchors(self.scales, self.ratios, self.stride,
                                             self.counts, self.levels, self.zero_area,
                                             self.feature_shapes, self.feature_strides)

    def generate_pyramid_anchors(self, scales, ratios, stride, counts, levels, zero_area, feature_shapes,
                                 feature_strides):
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
            anchors = self.generate_fmap_anchors(scales, ratios, stride, feature_shapes[0], feature_strides[0])

        elif levels > 1 and len(feature_shapes) > 1 and not parallel:
            # 将各层特征图上生成的anchors汇总在一起输出
            anchors = []
            for i in range(len(scales)):
                anchors.append(
                    self.generate_fmap_anchors(scales[i], ratios, stride, feature_shapes[i], feature_strides[i]))
            anchors = np.concatenate(anchors, axis=0)

        elif levels > 1 and len(feature_shapes) > 1 and parallel:
            # 将各层特征图上生成的anchors独立输出
            anchors = []
            for i in range(len(scales)):
                level_anchors = self.generate_fmap_anchors(scales[i], ratios, stride, feature_shapes[i],
                                                           feature_strides[i])
                level_anchors = Variable(torch.from_numpy(level_anchors).float(), requires_grad=False).cuda()
                anchors.append(level_anchors)
            return anchors

        else:
            raise ValueError('Error Pyramid Levels... %s' % levels)

        anchors = Variable(torch.from_numpy(anchors).float(), requires_grad=False).cuda()
        return [anchors]

    @staticmethod
    def generate_fmap_anchors(scales, ratios, anchor_stride, feature_shape, feature_stride):
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
        # 相对原图间隔为 anchor_stride*feature_stride  # TODO ???
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
        image_shape = np.array(
            [feature_shape[0], feature_shape[1], feature_shape[0], feature_shape[1]]) * feature_stride
        boxes = np.clip(boxes / image_shape.astype(np.float32), 0, 1)

        return boxes


def generate_anchors(config, method=None):
    if config.ANCHOR_METHOD == 'general':

        anchors_generate = GeneralAnchorLayer(scales=config.ANCHOR_SCALES,
                                              ratios=config.ANCHOR_ASPECTS,
                                              stride=config.ANCHOR_STRIDE,
                                              counts=config.ANCHORS_PER_IMAGE,
                                              levels=config.ANCHOR_LEVELS,
                                              zero_area=config.ANCHOR_ZERO_AREA,
                                              image_shape=config.IMAGE_SHAPE,
                                              feature_shapes=config.FUSION_SHAPES,
                                              feature_strides=config.FUSION_STRIDES)

    elif config.ANCHOR_METHOD == 'hotpoint':

        anchors_generate = HotAnchorLayer(scales=config.ANCHOR_SCALES,
                                          ratios=config.ANCHOR_ASPECTS,
                                          stride=config.ANCHOR_STRIDE,
                                          counts=config.ANCHORS_PER_IMAGE,
                                          levels=config.ANCHOR_LEVELS,
                                          method=config.ANCHOR_HEAT_METHOD,
                                          zero_area=config.ANCHOR_ZERO_AREA,
                                          image_shape=config.IMAGE_SHAPE,
                                          feature_shapes=config.FUSION_SHAPES,
                                          feature_strides=config.FUSION_STRIDES)

    else:
        raise ValueError('unknow confg.ANCHOR_METHOD %s' % config.ANCHOR_METHOD)

    return anchors_generate
