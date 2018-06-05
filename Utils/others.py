#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/4/9 9:28'
__author__ = 'ooo'

import torch


def pyramid_heat_filter(self, anchors, feature_maps):
    """
    # error: 当为多层金字塔特征图时，需要分别独立计算anchors在每张fmps上的热度值！！
    # 即：一个anchor是否是一个合适的proposal，由所有fmps进行同权投票
    # 某物体的anchor应该出现在金字塔各层，至少是多个层，或者至少非常大！
    # 只出现在一层上的anchor，将被忽略！
    """
    # anchors [A1, A2, ...]，[batch, count, (y1,x1,y2,x2)], t.Tensor
    # feature_maps [P1, P2, ...], [b, w, h, c], t.Tensor
    # heat_vals [batch, count, (heat_val)] b×c×1
    # all_heat_vals:  [heat_vals1, heat_vals2, heat_vals3, ...] → [batch, count, (heat_val_sum)]
    assert len(anchors) == len(feature_maps), '锚点和特征图的levels不相等！'
    assert anchors[0].shape[0] == feature_maps[0].shape[0], 'anchors与feature_map的batch维度不相等！'

    top_k = self.config.PROPOSALS_PER_IMAGE
    all_heat_vals = torch.zeros(anchors[0].shape[0:2] + (1,))
    for anchs, fmap in zip(anchors, feature_maps):
        heat_vals = self.anchor_heat_vals(anchs, fmap)
        # todo: error 各层特征图上的锚点并非是位置对应的！ 不能相加！
        all_heat_vals += heat_vals
    val, ind = torch.topk(all_heat_vals, k=top_k, dim=1)
    proposals = torch.gather(anchors, dim=1, index=ind.expand(ind.shape[0:2] + (anchors[0].size(-1),)))
    return proposals


@staticmethod
def anchor_heat_vals(anchors, feature_map):
    # 计算特征图上每个Anchor的领域平均热度
    # anchors [batch, count, (y1,x1,y2,x2)], torch.Tensor
    # feature_map [batch, count, h, w], torch.Tensor
    # anchors_heat [batch, count, (heat_val)], torch.Tensor
    # todo 锚点尺寸未进行转换
    assert anchors.shape[0] == feature_map.shape[0], 'anchors与feature_map的batch维度不相等！'

    # 计算热度图，方案1：各通道独立去均值 → 各通道独立求绝对值 → 再全通道求和
    fmap = torch.mean(torch.mean(feature_map, 2, True), 3, True)
    fmap = torch.sum(torch.abs(feature_map - fmap), 1)  # b*h*w

    # 映射归一化锚点到特征图尺寸
    fmap_h, fmap_w = fmap.size(1), fmap.size(2)
    scale = torch.from_numpy(np.array([fmap_h, fmap_w, fmap_h, fmap_w]))
    anchors *= scale

    # 遍历所有batches，遍历所有anchors
    anchors_heat = torch.zeros(anchors.shape[0:2] + (1,))
    for b in range(fmap.size(0)):
        bfmap = fmap[b, :, :]  # h*w
        banchors = anchors[b, :, :]  # N*4
        for a in range(banchors.shape[0]):
            anok = torch.round(banchors[a, :]).astype('int')
            bbox = bfmap[anok[0]:anok[2], anok[1]:anok[3]]
            heat = torch.sum(bbox) / bbox.numel()
            anchors_heat[b, a, 1] = heat

    return anchors_heat

