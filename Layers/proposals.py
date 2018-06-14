#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/6/6 6:14'
__author__ = 'ooo'

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
from Utils import utils


class HotProposalLayer(nn.Module):

    def __init__(self, counts, image_shape, levels):
        super(HotProposalLayer, self).__init__()
        self.counts = counts        # PROPOSALS_PER_IMAGE
        self.image_shape = image_shape
        self.levels = levels

    def forward(self, feature_maps, anchors):
        """
        # anchors [ANCHORS_PER_IMAGE, (y1, x1, y2, x2)], t.Tensor，已归一化
        # feature_maps [P1, P2, ...], P: (batch, h, w, c), t.Tensor
        # return: proposals [batch, PROPOSALS_PER_IMAGE, (y1, x1, y2, x2)]，t.Tensor，归一化
        # return: all_proposals [proposals1, proposals2, ...], 归一化
        """

        all_proposals = []

        # 单层特征图，单层锚点框，对应输出
        if self.levels == 1 and len(feature_maps) == 1 and len(anchors) == 1:
            proposals = self.simple_heat_filter(anchors[0], feature_maps[0], self.counts)
            all_proposals.append(proposals)  # shape: [[batch, top-k, (y1, x1, y2, x2)]]

        # 多层特征图，单层锚点框，综合输出
        elif self.levels > 1 and len(feature_maps) > 1 and len(anchors) == 1:
            proposals = self.pyramid_heat_filter(anchors[0], feature_maps, self.counts)
            all_proposals.append(proposals)

        # 多层特征图，多层锚点框，并行输出
        elif self.levels > 1 and len(feature_maps) > 1 and len(anchors) > 1:
            for fmap, anch in zip(feature_maps, anchors):
                proposals = self.simple_heat_filter(anch, fmap, self.counts)
                all_proposals.append(proposals)
        else:
            raise ValueError('错误的特征层级数！')

        return all_proposals

    @staticmethod
    def simple_heat_filter(anchors, feature_map, counts):
        """
        # 返回单张特征图上的锚点热度值
        # anchors： [N, (y1, x1, y2, x2)], t.Tensor，归一化
        # feature_map： [batch, channel, h, w], t.Tensor
        # return: proposals [batch, counts, (y1, x1, y2, x2)]
        """
        anchors = anchors.data
        feature_map = feature_map.data

        # 计算热度图，方案1：各通道独立去均值 → 各通道独立求绝对值 →  map(0, 1)? →  再全通道求和
        # fmap shape: batch x h x w
        fmap = torch.mean(torch.mean(feature_map, 2, True), 3, True)
        fmap = torch.abs(feature_map - fmap)
        fmap = torch.sum(fmap, 1)
        feature_map = None

        # 映射归一化锚点到特征图尺寸的锚点
        stride = torch.from_numpy(np.array([fmap.size(1), fmap.size(2), fmap.size(1), fmap.size(2)])).float().cuda()
        anchors *= stride

        # 遍历所有batches的fmaps，遍历所有anchors, 计算每个anchor内的热度
        anchors_heat = torch.zeros(fmap.shape[0], anchors.shape[0]).cuda()  # [batch, N, (heat_val)]
        for b in range(fmap.size(0)):
            bfmap = fmap[b, :, :]  # h*w
            for a in range(anchors.shape[0]):
                anok = torch.round(anchors[a, :]).int()
                bbox = bfmap[anok[0]:anok[2], anok[1]:anok[3]]
                heat = torch.sum(bbox) / bbox.numel()
                anchors_heat[b, a] = heat

        # 过滤锚点框内的热度值
        _, index = torch.topk(anchors_heat, k=counts, dim=1)
        # index: [batch, top_k] -> [batch, top_k, 1] → [batch, top_k, 4]
        index = index.unsqueeze(-1).expand(index.shape + (anchors.shape[-1],)).cuda()
        # anchors: [N, 4] -> [1, N, 4] -> [batch, N, 4]
        anchors = anchors.unsqueeze(0).expand((index.shape[0],) + anchors.shape)
        proposals = torch.gather(anchors, dim=1, index=index)

        # 映射特征图尺寸的锚点回归一化尺寸
        proposals /= stride

        proposals = Variable(proposals, requires_grad=False)
        return proposals

    @staticmethod
    def pyramid_heat_filter(anchors, feature_maps, counts):
        """
        # 特征图P上的某个anchor是否能成为proposal, score = Σ(α*φ(P[anchor])), φ是某种测量，比如热度测量，轮廓数测量。
        # anchors： [N, (y1, x1, y2, x2)], t.Tensor，归一化
        # 将特征图[P1,P2,P3,P4,P5]处理为热度图[H1,H2,H3,H4,H5], [batch, h, w]
        # 计算热度图，方案1：各通道独立去均值 → 各通道独立求绝对值 →  map(0, 1)? →  再全通道求和
        # anchors_heat: [batch_nums, anchor_nums, (heat_val,)*fmap_nums]
        # return: proposals [batch, counts, (y1, x1, y2, x2)]，归一化值
        """
        anchors = anchors.data
        anchors_num = anchors.shape[0]
        batches = feature_maps[0].shape[0]

        assert anchors_num > counts, '总锚点数量不能少于预抓取锚点数量'

        heat_maps = []
        for i, fmap in enumerate(feature_maps):
            fmap_mean = torch.mean(torch.mean(fmap.data, 2, True), 3, True)
            fmap_mean = torch.abs(fmap.data - fmap_mean)
            fmap_mean = torch.sum(fmap_mean, 1)     # shape [b, h, w]
            heat_maps.append(fmap_mean)

        anchors_heat = torch.zeros((batches, anchors_num, len(heat_maps))).cuda()
        for b in range(batches):
            for c, hmap in enumerate(heat_maps):
                hmap = hmap[b, :, :]
                stride = torch.FloatTensor([hmap.size(0), hmap.size(1), hmap.size(0), hmap.size(1)]).cuda()
                sanchors = torch.round(anchors*stride).int()
                for a in range(anchors_num):
                    anok = sanchors[a, :]
                    bbox = hmap[anok[0]:anok[2], anok[1]:anok[3]]
                    heat = torch.sum(bbox) / bbox.numel()
                    anchors_heat[b, a, c] = heat

        anchors_heat = torch.sum(anchors_heat, dim=2)                      # [b, N]
        _, index = torch.sort(anchors_heat, dim=1, descending=True)        # [b, N]
        anchors = anchors.unsqueeze(0).expand((batches,) + anchors.shape)  # [b, N, 4]
        index = index[:, 0:counts]
        proposals = torch.gather(anchors, dim=1, index=index.unsqueeze(-1).expand(index.shape + anchors.shape[-1]))
        proposals = Variable(proposals, requires_grad=False)
        return proposals
