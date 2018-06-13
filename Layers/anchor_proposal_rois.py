#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/4/3 22:51'
__author__ = 'ooo'

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
from Utils import utils
from RoiTransforms.roi_align2.crop_and_resize import CropAndResizeFunction


class HotAnchorLayer(nn.Module):
    def __init__(self, scales, ratios, counts, image_shape, level_nums, heat_method, zero_area):
        # scales 按大到小排列
        """
        :param scales:
        :param ratios:
        :param counts:
        :param image_shape: 原图尺寸 (h,w,c)
        :param fmap_shapes: 特征图尺寸
        :param fmap_scales: 特征图尺寸相对于原图尺寸的缩放倍数，2^n
        :param level_nums:
        :param heat_method:
        :param zero_area:
        """
        super(HotAnchorLayer, self).__init__()
        self.scales = scales
        self.ratios = ratios
        self.counts = counts
        self.shape = image_shape
        self.level_nums = level_nums
        self.heat_method = heat_method
        self.zero_area = zero_area

    def forward(self, x):
        """
        x: [fmap1, fmap2, ...], 尺寸按大到小排列
        fmap1: [batch, c, h, w]，tensor
        anchors: [batch, N, (y1, x1, y2, x2)] , array
        all_anchors: [anchors1, anchors2, ...], tensor
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

        all_anchors = []

        # Note：单层/多层情况下，每个点上的scale不同，anchors中的数量N不相同！
        # all_anchors : counts*len(scales)*len(ratios)*level_nums
        # 1000*3*3*1  vs  1000*1*3*5

        # 单层特征图
        if self.level_nums == 1:
            centers = heat_centers(x[0], self.counts, self.shape)
            anchors = self.boxes_generate(centers, self.scales, self.ratios, self.shape, x[0].shape[2:], self.zero_area)
            all_anchors.append(anchors)

        # 金字塔特征图
        elif self.level_nums > 1:
            assert self.level_nums == len(self.scales), '特征层levels与scales不相同！'
            for i, fmap in enumerate(x):
                centers = heat_centers(fmap, self.counts, self.shape)
                anchors = self.boxes_generate(centers, self.scales[i], self.ratios, self.shape, fmap.shape[2:], self.zero_area)
                all_anchors.append(anchors)
        else:
            raise ValueError('错误的特征融合层数！%s')

        return all_anchors

    @staticmethod
    def accumulate_heat_map(x, counts, shape):
        """方案3
        累积最大热度：将所有通道独立去均值，独立求绝对值，再映射到01？，再累积相加，然后比大小.
        返回的centers，必须是以原图尺寸为参照！
        :param  x: [batch, channels, height, weight]，float32
        :param  shape: (h, w, c) 原图尺寸
        :return: centers: [batch, N, (y, x)] tensor, 原图尺寸为参照
        """
        assert x.size(2) * x.size(3) > counts, 'need anchors counts > pixels of fmap'
        strides = torch.FloatTensor([shape[0] / x.size(2), shape[1] / x.size(3)])
        centers = x.data.new(x.size(0), counts, 2).zero_()
        for b in range(x.size(0)):
            fmap = x[b, :, :, :]
            fmap = torch.sum(torch.abs(fmap - torch.mean(torch.mean(fmap, 1, True), 2, True)), 0)
            kval, _ = torch.topk(fmap.view(-1), k=counts)
            # points = torch.where(fmap >= kval[-1])
            # points = torch.cat([points[0].contiguous().view(-1, 1), points[1].contiguous().view(-1, 1)], dim=1)
            points = np.where((fmap >= kval[-1]).data)
            points_y = torch.FloatTensor(points[0]).contiguous()
            points_x = torch.FloatTensor(points[1]).contiguous()
            points = torch.cat([points_y.view(-1, 1), points_x.view(-1, 1)], dim=1)
            centers[b, :, :] = points[0: counts] * strides
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
        scales, centers 都以原图尺寸为参照，因此生成的盒子最后/shape, 即可归一化！

        ??? todo: how to filter zero_area

        scales = array, [32, 64, 128, 256]，levels>1时，每层分配1个scale
        ratios = array, [0.5, 1, 2]
        centers = tensor, [batch, N, (y, x)]
        :all_boxes = tensor, [batch, N*len(scales)*len(ratios), (y1, x1, y2, x2)]
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
        scales, ratios = scales.flatten(), ratios.flatten()
        # 按ratios调整scales尺度
        heights = (scales / np.sqrt(ratios)).reshape(-1, 1)
        widths = (scales * np.sqrt(ratios)).reshape(-1, 1)

        for b in range(batches):
            # 第二步：获取y/x  shape: [N2, (y, x)]
            # box_centers = centers[b, :, :]
            # ctr_y, ctr_x = centers[b, :, 0], centers[b, :, 1]

            box_heights, box_centers_y = np.meshgrid(heights, centers[b, :, 0])
            box_widths, box_centers_x = np.meshgrid(widths, centers[b, :, 1])

            # Reshape to get a list of (y, x) and a list of (h, w)
            box_centers = np.stack([box_centers_y, box_centers_x], axis=2)
            box_sizes = np.stack([box_heights, box_widths], axis=2)
            box_centers = np.reshape(box_centers, (-1, 2))
            box_sizes = np.reshape(box_sizes, (-1, 2))

            # 第四步：中心坐标+长宽 变换为 对角顶点坐标
            # Convert to corner coordinates (y1, x1, y2, x2)
            # np.concatenate([[y1, x1],[y2, x2]], axis=1) → [y1, x1, y2, x2]
            all_boxes[b, :, :] = np.concatenate([box_centers - 0.5 * box_sizes,
                                                 box_centers + 0.5 * box_sizes], axis=1)

        # 归一化
        # [batch, counts, (y1, x1, y2, x2)]
        all_boxes /= np.array([image_shape[0], image_shape[1], image_shape[0], image_shape[1]]).astype(np.float32)

        # crop tp [0~1] within image shape
        all_boxes = torch.from_numpy(all_boxes).cuda().clamp(0, 1)

        # # filter 0 area boxes, object must have 4*4 pixels on original image
        # # note : each batch has different counts of zero-area boxes
        # zero_area = zero_area/((image_shape[0]/fmap_shape[0])*(image_shape[1]/fmap_shape[1]))   # 0.015625   1024/32
        # boxes_area = (all_boxes[:, :, 2] - all_boxes[:, :, 0])*(all_boxes[:, :, 3] - all_boxes[:, :, 1])  # shape [b, counts]
        # boxes_index = []
        # [boxes_index.append(np.where(boxes_area[b, :] > zero_area)[0]) for b in range(boxes_area.size(0))]
        # minum_index = [len(ind) for ind in boxes_index]
        # minum_index.sort()
        # minum_index = minum_index[0]
        # boxes_index = [torch.LongTensor(ind[:minum_index]).unsqueeze(0) for ind in boxes_index]
        # boxes_index = torch.cat(boxes_index, dim=0).unsqueeze(-1).expand(all_boxes.size(0), minum_index, all_boxes.size(-1))
        # # shape: [b, same_counts] -> [b, same_counts, 1] -> [b, same_counts, 4]
        # all_boxes = torch.gather(all_boxes, dim=1, index=boxes_index.cuda())

        all_boxes = Variable(all_boxes, requires_grad=False)
        return all_boxes


class HotProposalLayer(nn.Module):
    def __init__(self, mode, counts, image_shape, levels):
        """
        1. 利用bbox内热度过滤anchors
        :param mode:
        :param counts:
        :param shape: 原图尺寸 (h, w, c)
        :param level:
        """
        self.mode = mode
        self.counts = counts
        self.shape = image_shape
        self.levels = levels
        super(HotProposalLayer, self).__init__()

    def forward(self, anchors, feature_maps):
        # anchors [A1, A2, ...]，A：[batch, counts, (y1, x1, y2, x2)], t.Tensor，已归一化
        # feature_maps [P1, P2, ...], P: (batch, h, w, c), t.Tensor
        # proposals [batch, counts, (y1, x1, y2, x2)]，t.Tensor，归一化
        # all_proposals [proposals1, proposals2, ...], 归一化

        all_proposals = []

        # 单层特征图
        if self.levels == 1:
            proposals = self.simple_heat_filter(anchors[0], feature_maps[0], self.counts)
            all_proposals.append(proposals)

        # 多层特征图
        # 1.每层特征图独立处理/独立排序, 独立返回
        # 2.各层特征图独立处理/独立排序, 综合返回，同一个点可能同时在不同层上被检测出！
        elif self.levels > 1:
            for anchs, fmap in zip(anchors, feature_maps):
                proposals = self.simple_heat_filter(anchs, fmap, self.counts)
                all_proposals.append(proposals)
                # todo all proposals handle
                # overall handle, so can PyramidROIAlign
        else:
            raise ValueError('错误的特征层级数！')

        return all_proposals

    @staticmethod
    def simple_heat_filter(anchors, feature_map, counts):
        # 返回单个level的特征图上的锚点热度值
        # anchors [batch, count, (y1, x1, y2, x2)], t.Tensor，归一化
        # feature_map [batch, channel, h, w], t.Tensor
        # anchors_heat [batch, count, heat_val], t.Tensor
        anchors = anchors.data
        feature_map = feature_map.data

        assert anchors.shape[0] == feature_map.shape[0], 'anchors与feature_map的batch维度不相等！'

        # 计算热度图，方案1：各通道独立去均值 → 各通道独立求绝对值 →  map(0, 1)? →  再全通道求和
        fmap = torch.mean(torch.mean(feature_map, 2, True), 3, True)
        fmap = torch.sum(torch.abs(feature_map - fmap), 1)  # b*h*w

        # 映射归一化锚点到特征图尺寸的锚点
        stride = torch.FloatTensor([fmap.size(1), fmap.size(2), fmap.size(1), fmap.size(2)]).cuda()
        anchors *= stride

        # 遍历所有batches，遍历所有anchors
        anchors_heat = torch.zeros(anchors.shape[0:2]).cuda()  # [batch, N, (heat_val)]
        for b in range(fmap.size(0)):
            bfmap = fmap[b, :, :]  # h*w
            banch = anchors[b, :, :]  # N*4
            for a in range(banch.shape[0]):
                anok = torch.round(banch[a, :]).int()
                bbox = bfmap[anok[0]:anok[2], anok[1]:anok[3]]
                heat = torch.sum(bbox) / bbox.numel()
                anchors_heat[b, a] = heat

        # 过滤锚点框内的热度值
        _, ind = torch.topk(anchors_heat, k=counts, dim=1)
        # ind: [b, top_k] -> [b, top_k, 1] → [b, top_k, 4]
        index = ind.unsqueeze_(-1).expand(ind.shape[0:2] + (anchors.size(-1),)).cuda()
        proposals = torch.gather(anchors, dim=1, index=index)

        # 映射特征图尺寸的锚点回归一化尺寸
        proposals /= stride

        proposals = Variable(proposals, requires_grad=False)
        return proposals


class RoiTargetLayer(nn.Module):
    def __init__(self, config):
        """
        TRAIN_ROIS_PER_IMAGE, PROPOSALS_PER_IMAGE, MAX_GT_INSTANCES
        :param config:
        """
        self.config = config
        super(RoiTargetLayer, self).__init__()

    def forward(self, proposals, gt_class_ids, gt_boxes, gt_masks):
        """
        输入：
        proposals: [[batch, N, (y1, x1, y2, x2)], ., ...] 归一化，零填充. tensor
        gt_class_ids: [batch, MAX_GT_INSTANCES] 类标. tensor
        gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] 归一化. tensor
        gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type. tensor
        输出：
        rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)], [rois1, rois2 ...] 归一化，零填充. tensor
        """
        if self.config.FEATURE_FUSION_LEVELS == 1:

            proposals = proposals[0]
            rois = []
            target_class_ids = []
            target_deltas = []
            target_masks = []
            for b in range(proposals.size(0)):
                brois, btarget_class_ids, btarget_deltas, btarget_masks = \
                    self.detection_targets_graph(proposals[b, :, :],
                                                 gt_class_ids[b, :],
                                                 gt_boxes[b, :, :],
                                                 gt_masks[b, :, :, :],
                                                 config=self.config)
                rois.append(brois.unsqueeze(0))
                target_class_ids.append(btarget_class_ids.unsqueeze(0))
                target_deltas.append(btarget_deltas.unsqueeze(0))
                target_masks.append(btarget_masks.unsqueeze(0))

            rois = torch.cat(rois, dim=0)
            target_class_ids = torch.cat(target_class_ids, dim=0)
            target_deltas = torch.cat(target_deltas, dim=0)
            target_masks = torch.cat(target_masks, dim=0)

            return rois, target_class_ids, target_deltas, target_masks

        elif self.config.FEATURE_FUSION_LEVELS > 1:

            all_proposals = proposals
            all_rois = []
            all_target_class_ids = []
            all_target_deltas = []
            all_target_masks = []

            for i, proposals in enumerate(all_proposals):

                rois = []
                target_class_ids = []
                target_deltas = []
                target_masks = []
                for b in range(proposals.size(0)):
                    brois, btarget_class_ids, btarget_deltas, btarget_masks = \
                        self.detection_targets_graph(proposals[b, :, :],
                                                     gt_class_ids[b, :],
                                                     gt_boxes[b, :, :],
                                                     gt_masks[b, :, :, :],
                                                     config=self.config)
                    rois.append(brois.unsqueeze(0))
                    target_class_ids.append(btarget_class_ids.unsqueeze(0))
                    target_deltas.append(btarget_deltas.unsqueeze(0))
                    target_masks.append(btarget_masks.unsqueeze(0))

                all_rois.append(torch.cat(rois, dim=0))
                all_target_class_ids.append(torch.cat(target_class_ids, dim=0))
                all_target_deltas.append(torch.cat(target_deltas, dim=0))
                all_target_masks.append(torch.cat(target_masks, dim=0))

            return all_rois, all_target_class_ids, all_target_deltas, all_target_masks

        else:
            raise ValueError('错误的特征级数！！')

    @staticmethod
    def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
        """
        # 匹配流程
        1. 移除 proposal的 0填充
        2. 区分 拥挤和非拥挤 GT-boxes
        3. 计算 Overlaps, proposals, gt_boxes, 2D表格内进行竖横筛选，先筛选垂直方向proposal, 再筛选水平方向gtboxes
        4. 正负 RoIs 判定，IoU, counts, ratio
        5、为正 ROIs 配置 gt-boxes, gt-class-ids targets
        6、计算 ROIs 的校正量 deltas targets
        7. 计算 mask targets
        8. 组合 正负RoIs

        # 匹配标准
        1。与某个GT的 IoU >= 0.5为正，与所有GT的 IoU < 0.5为负
        2。存在一个GT Iou >= 0.7 为正， 与所有GT的 IoU < 0.5为负， 介于0.5~0.7之间为中性
        3。与每个GT最近的那个ROI, 必须为正, 即每个GT都必须要有匹配对象

        输入:
        proposals: [N, (y1, x1, y2, x2)] 归一化，零填充.
        gt_class_ids: [MAX_GT_INSTANCES] 类标.
        gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] 归一化.
        gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

        返回:
            Target ROIs and corresponding class IDs, bounding box shifts, and masks.
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
        deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
                Class-specific bbox refinments. bbox的偏移量是按类指定的。
        masks: [TRAIN_ROIS_PER_IMAGE, height, width). Masks cropped to bbox
               boundaries and resized to neural network output size.
               Masks被按照bbox裁切，再缩放到config中指定的输出大小

        Note: 如果没有足够的target ROIs，会进行零填充.
              MAX_GT_INSTANCES < TRAIN_ROIS_PER_IMAGE
        """

        assert proposals.size(0) > 0, '当前的proposal是空的！'

        # gt_class_ids, gt_boxes, gt_masks = gt_class_ids.data, gt_boxes.data, gt_masks.data

        # 1. 移除proposals，GT中的0填充

        # 2. 区分拥挤GT & 不拥挤GT

        # 3、计算proposals和gt_boxes的Overlaps

        # Compute overlaps matrix [proposals, gt_boxes]
        overlaps = utils.bbox_overlaps(proposals, gt_boxes)  # shape: N×K

        # 4、判定正负ROIs
        # Determine postive and negative ROIs

        # method1: given counts, max_min_threshold, firstly filter by max_min, then both randomly select P/N-rois
        # method2: given counts, firstly select top-k(p_counts) as P-rois, others randomly selecte n_counts as N-rois
        # method1 can always keep hold the overlap > max in P-rois, overlap < min in N-rois.
        # method2 can always keep hold the max overlap proposals, only N-rois is random!! don't need setting max_min

        method1, method2 = (False, True)

        if method1:
            # dim1 : 每个proposal的最佳gt_box的iou值
            roi_iou_max = torch.max(overlaps, dim=1)[0].data  # shape: N
            # 4.1. Positive ROIs are those with >= 0.5 IoU with a GT box
            positive_indices = torch.from_numpy(
                np.where((roi_iou_max >= config.ANCHOR_GTBOX_IOU[0]))[0]).long().cuda()  # shape: N
            # 4.2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
            # negative_indices = np.where(np.logical_and(roi_iou_max.numpy() < 0.5, no_crowd_bool.numpy()))[0]
            # negative_indices = torch.where((roi_iou_max < 0.5).int().__and__(no_crowd_bool))[0]
            roi_iou_sort = torch.sort(roi_iou_max, dim=0)
            negative_indices = torch.from_numpy(np.where(roi_iou_max < config.ANCHOR_GTBOX_IOU[1])[0]).long().cuda()

            # Subsample ROIs. Aim for 33% positive
            # todo ?? randomly subsample is ok ??
            # Positive ROIs 在所有正ROIs中，随机选择config中指定的数量个
            # 实际数量有可能不足config中的设定值，因此最终的数量由shape[0]推算而出！
            positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROIS_POSITIVE_RATIO)  # all*pr
            positive_indices = positive_indices[list(torch.randperm(positive_indices.numel()))][:positive_count]
            positive_count = positive_indices.shape[0]  # 切片允许索引超限，因此实际数量仍需切片后统计
            # Negative ROIs. Add enough to maintain positive:negative ratio.
            # 最终的ROIs数量，必须满足预设的正负比例，但不一定能同时满足预设的总数量
            r = 1.0 / config.ROIS_POSITIVE_RATIO  # 1/0.33
            negative_count = math.floor((r - 1) * positive_count)  # total*pr*(1/pr-1)=all*(1-pr)
            negative_indices = negative_indices[list(torch.randperm(negative_indices.numel()))][:negative_count]

        elif method2:
            roi_iou_max = torch.max(overlaps, dim=1)[0].data  # shape: N
            roi_iou_ind = torch.sort(roi_iou_max, dim=0, descending=True)[1]
            positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROIS_POSITIVE_RATIO)
            negative_count = config.TRAIN_ROIS_PER_IMAGE - positive_count
            positive_indices = roi_iou_ind[0: positive_count]
            negative_indices = roi_iou_ind[positive_count:]
            negative_indices = negative_indices[torch.randperm(negative_indices.numel())[0: negative_count].cuda()]

        else:
            raise ValueError('wrongt method!')

        # Gather selected ROIs 收集正负ROIs
        # index shape : N → N×1 → N×4
        index = Variable(positive_indices.unsqueeze(-1).expand(positive_indices.size(0), proposals.size(-1)),
                         requires_grad=False)
        positive_rois = torch.gather(proposals, dim=0, index=index)
        index = Variable(negative_indices.unsqueeze(-1).expand(negative_indices.size(0), proposals.size(-1)),
                         requires_grad=False)
        negative_rois = torch.gather(proposals, dim=0, index=index)

        # 5. 为正ROIs分配 GT-boxes + GT-class-ids

        # Assign positive ROIs to GT boxes. 沿着gtbox方向(dim=1)统计最大值，得到每个anchor所对应的最大gtbox的索引
        # index shape : N → N×1 → N×gt_boxes_count
        index = Variable(positive_indices.unsqueeze(-1).expand(positive_indices.size(0), overlaps.size(-1)),
                         requires_grad=False)
        positive_overlaps = torch.gather(overlaps, dim=0, index=index)
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1].long()  # N
        # N → N×1 → N×4
        index = roi_gt_box_assignment.unsqueeze(-1).expand(roi_gt_box_assignment.size(0), gt_boxes.size(-1))
        roi_gt_boxes = torch.gather(gt_boxes, dim=0, index=index)
        roi_gt_class_ids = torch.gather(gt_class_ids, dim=0, index=roi_gt_box_assignment)

        # 6、计算ROIs的校正量deltas

        # Compute bbox refinement for positive ROIs # 对正ROIs计算bbox的修正量
        deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
        deltas /= Variable(torch.FloatTensor(config.BBOX_STD_DEV).cuda(), requires_grad=False)

        # 6、抓取正ROI的-masks, 并计算mask targets

        # 在原始的GT-mask上，裁切位于roi_box中的那部分mask出来，再缩放到指定shape大小。
        # GT-mask是一个H×W的二值图，因此裁切出来的仍然是一个小二值图
        # 此小二值图，即是此roi_box的gt_mask_targets，可用于计算二值交叉熵损失
        # Assign positive ROIs to GT masks
        # Permute masks from [h, w, n] to [N, height, width, channel==1] 跟box的坐标相对应
        transposed_masks = torch.unsqueeze(torch.transpose(gt_masks, 2, 0, 1), dim=-1)

        # Pick the right mask for each ROI
        roi_masks_index = roi_gt_box_assignment.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        roi_masks_index = roi_masks_index.expand((roi_gt_box_assignment.size(0),) + transposed_masks.shape[1:])
        roi_masks = torch.gather(transposed_masks, dim=0, index=roi_masks_index)
        # Compute mask targets, 挖出与roi相对应的mask
        boxes = positive_rois
        if config.USE_MINI_MASK:
            # Transform ROI corrdinates from normalized image space
            # to normalized mini-mask space.
            y1, x1, y2, x2 = torch.split(positive_rois, 1, dim=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = torch.split(roi_gt_boxes, 1, dim=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = torch.cat([y1, x1, y2, x2], 1)
        box_ids = Variable(torch.arange(roi_masks.size()[0]), requires_grad=False).int().cuda()
        # 从roi_masks中切出boxes，再resize到config.MASK_SHAPE大小
        # masks = torch.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes, box_ids, config.MASK_SHAPE)
        crfuc = CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)
        masks = Variable(crfuc(roi_masks.unsqueeze(1), boxes, box_ids).data, requires_grad=False)

        # Remove the extra dimension from masks.
        # transposed的时候增加了一维度 expand_dims(*,-1,*,*) -> shape Nx28x28
        masks = torch.squeeze(masks, dim=1)
        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss. 应用二值交叉熵损失前，进行round处理
        masks = torch.round(masks)  # shape Nx28x28

        # 7、组合正负ROIs，并进行零填充

        # Append negative ROIs and pad bbox deltas and masks that
        # are not used for negative ROIs with zeros.
        zero_padded = False
        if zero_padded:
            rois = torch.cat([positive_rois, negative_rois], dim=0)  # shape Nx4
            N = negative_rois.size(0)
            P = config.TRAIN_ROIS_PER_IMAGE - rois.size(0)
            P = P if P > 0 else 0
            rois = F.pad(rois, (0, 0, 0, P))
            # roi_gt_boxes = F.pad(roi_gt_boxes, (0, 0, 0, N + P))
            roi_gt_class_ids = F.pad(roi_gt_class_ids, (0, N + P))
            deltas = F.pad(deltas, (0, 0, 0, N + P))
            masks = F.pad(masks, (0, 0, 0, 0, 0, N + P))
        else:
            rois = torch.cat([positive_rois, negative_rois], dim=0)  # shape Nx4
            N = negative_rois.size(0)
            P = config.TRAIN_ROIS_PER_IMAGE - rois.size(0)
            P = P if P > 0 else 0
            rois = F.pad(rois, (0, 0, 0, P))
            # roi_gt_boxes = F.pad(roi_gt_boxes, (0, 0, 0, N + P))
            roi_gt_class_ids = F.pad(roi_gt_class_ids, (0, N + P))
            deltas = F.pad(deltas, (0, 0, 0, N + P))
            masks = F.pad(masks, (0, 0, 0, 0, 0, N + P))
        return rois, roi_gt_class_ids, deltas, masks

    @staticmethod
    def detection_targets_graph_nocrowd(proposals, gt_class_ids, gt_boxes, gt_masks, config):
        """
        # 匹配流程
        1. 移除 proposal的 0填充
        2. 区分 拥挤和非拥挤 GT-boxes
        3. 计算 Overlaps, proposals, gt_boxes, 2D表格内进行竖横筛选，先筛选垂直方向proposal, 再筛选水平方向gtboxes
        4. 正负 RoIs 判定，IoU, counts, ratio
        5、为正 ROIs 配置 gt-boxes, gt-class-ids targets
        6、计算 ROIs 的校正量 deltas targets
        7. 计算 mask targets
        8. 组合 正负RoIs

        # 匹配标准
        1。与某个GT的 IoU >= 0.5为正，与所有GT的 IoU < 0.5为负
        2。存在一个GT Iou >= 0.7 为正， 与所有GT的 IoU < 0.5为负， 介于0.5~0.7之间为中性
        3。与每个GT最近的那个ROI, 必须为正, 即每个GT都必须要有匹配对象

        输入:
        proposals: [N, (y1, x1, y2, x2)] 归一化，零填充.
        gt_class_ids: [MAX_GT_INSTANCES] 类标.
        gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] 归一化.
        gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

        返回:
            Target ROIs and corresponding class IDs, bounding box shifts, and masks.
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
        deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
                Class-specific bbox refinments. bbox的偏移量是按类指定的。
        masks: [TRAIN_ROIS_PER_IMAGE, height, width). Masks cropped to bbox
               boundaries and resized to neural network output size.
               Masks被按照bbox裁切，再缩放到config中指定的输出大小

        Note: 如果没有足够的target ROIs，会进行零填充.
              MAX_GT_INSTANCES < TRAIN_ROIS_PER_IMAGE
        """

        assert proposals.size(0) > 0, '当前的proposal是空的！'

        gt_class_ids, gt_boxes, gt_masks = gt_class_ids.data, gt_boxes.data, gt_masks.data

        # 1. 移除proposals，GT中的0填充

        # 2. 区分拥挤GT & 不拥挤GT
        # Handle COCO crowds
        # A crowd box in COCO is a bounding box around several instances. Exclude
        # them from training. A crowd box is given a negative class ID.
        # 当一个BOX围住好几个物体实例时，称为Crowdbox，将其从训练阶段排除，给予一个负标签
        crowd_ix = torch.from_numpy(np.where(gt_class_ids < 0)[0]).long()
        non_crowd_ix = torch.from_numpy(np.where(gt_class_ids > 0)[0]).long()

        # crowd_gt_class_ids = torch.gather(gt_class_ids, index=crowd_ix)
        crowd_boxes = torch.gather(gt_boxes, dim=0, index=crowd_ix.unsqueeze(-1).expand(crowd_ix.size(0), 4))
        # crowd_ix_mask = crowd_ix.unsqueeze(0).unsqueeze(0).expand(gt_masks.shape[0:2] + (crowd_ix.size(0),))
        # crowd_masks = torch.gather(gt_masks, dim=2, index=crowd_ix_mask)

        gt_class_ids = torch.gather(gt_class_ids, index=non_crowd_ix)
        gt_boxes = torch.gather(gt_boxes, index=non_crowd_ix.unsqueeze(-1).expand(non_crowd_ix.size(0), 4))
        no_crowd_ix_mask = non_crowd_ix.unsqueeze(0).unsqueeze(0).expand(gt_masks.shape[0:2] + (non_crowd_ix.size(0),))
        gt_masks = torch.gather(gt_masks, dim=2, index=no_crowd_ix_mask)

        # 3、计算proposals和gt_boxes的Overlaps

        # Compute overlaps matrix [proposals, gt_boxes]
        overlaps = utils.bbox_overlaps(proposals, gt_boxes)  # shape: N×K
        crowd_overlaps = utils.bbox_overlaps(proposals, crowd_boxes)
        crowd_iou_max = torch.max(crowd_overlaps, dim=1)[0]
        no_crowd_bool = (crowd_iou_max < 0.001)

        # 4、判定正负ROIs

        # Determine postive and negative ROIs
        # dim1 : 每个proposal的最佳gt_box的iou值
        roi_iou_max = torch.max(overlaps, dim=1)[0]  # shape: N
        # 4.1. Positive ROIs are those with >= 0.5 IoU with a GT box
        positive_indices = torch.from_numpy(np.where((roi_iou_max >= 0.5))[0]).long()  # shape: N
        # 4.2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
        # negative_indices = np.where(np.logical_and(roi_iou_max.numpy() < 0.5, no_crowd_bool.numpy()))[0]
        # negative_indices = torch.where((roi_iou_max < 0.5).int().__and__(no_crowd_bool))[0]
        negative_indices = torch.from_numpy(np.where(np.logical_and(roi_iou_max < 0.5, no_crowd_bool))[0]).long()

        # Subsample ROIs. Aim for 33% positive
        # Positive ROIs 在所有正ROIs中，随机选择config中指定的数量个
        # 实际数量有可能不足config中的设定值，因此最终的数量由shape[0]推算而出！
        positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROIS_POSITIVE_RATIO)  # all*pr
        positive_indices = positive_indices[torch.randperm(positive_indices.numel())][:positive_count]
        positive_count = positive_indices.shape[0]  # 切片允许索引超限，因此实际数量仍需切片后统计
        # Negative ROIs. Add enough to maintain positive:negative ratio.
        # 最终的ROIs数量，必须满足预设的正负比例，但不一定能同时满足预设的总数量
        r = 1.0 / config.ROIS_POSITIVE_RATIO  # 1/0.33
        negative_count = math.floor((r - 1) * positive_count)  # total*pr*(1/pr-1)=all*(1-pr)
        negative_indices = negative_indices[torch.randperm(negative_indices.numel())][:negative_count]
        # Gather selected ROIs 收集正负ROIs
        positive_rois = torch.gather(proposals, dim=0,  # N → N×1 → N×4
                                     index=positive_indices.unsqueeze(-1).expand(positive_indices.size(0), 4))
        negative_rois = torch.gather(proposals, dim=0,
                                     index=negative_indices.unsqueeze(-1).expand(positive_indices.size(0), 4))

        # 5. 为正ROIs分配 GT-boxes + GT-class-ids

        # Assign positive ROIs to GT boxes. 沿着gtbox方向(dim=1)统计最大值，得到每个anchor所对应的最大gtbox的索引
        positive_overlaps = torch.gather(overlaps, dim=0,  # N → N×1 → N×4
                                         index=positive_indices.unsqueeze(-1).expand(positive_indices.size(0), 4))
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1].long()  # N
        roi_gt_boxes = torch.gather(gt_boxes, dim=0,  # N → N×1 → N×4
                                    index=roi_gt_box_assignment.unsqueeze(-1).expand(roi_gt_box_assignment.size(0), 4))
        roi_gt_class_ids = torch.gather(gt_class_ids, dim=0, index=roi_gt_box_assignment)

        # 6、计算ROIs的校正量deltas

        # Compute bbox refinement for positive ROIs # 对正ROIs计算bbox的修正量
        deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
        deltas /= config.BBOX_STD_DEV

        # 6、抓取正ROI的-masks, 并计算mask targets

        # 在原始的GT-mask上，裁切位于roi_box中的那部分mask出来，再缩放到指定shape大小。
        # GT-mask是一个H×W的二值图，因此裁切出来的仍然是一个小二值图
        # 此小二值图，即是此roi_box的gt_mask_targets，可用于计算二值交叉熵损失

        # Assign positive ROIs to GT masks
        # Permute masks to [N, height, width, channel==1] 跟box的坐标相对应
        transposed_masks = torch.unsqueeze(torch.transpose(gt_masks, 2, 0, 1), -1)
        # Pick the right mask for each ROI
        roi_masks_index = roi_gt_box_assignment.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        roi_masks_index = roi_masks_index.expand((roi_gt_box_assignment.size(0),) + transposed_masks.shape[1:])
        roi_masks = torch.gather(transposed_masks, dim=0, index=roi_masks_index)
        # Compute mask targets, 挖出与roi相对应的mask
        boxes = positive_rois
        if config.USE_MINI_MASK:
            # Transform ROI corrdinates from normalized image space
            # to normalized mini-mask space.
            y1, x1, y2, x2 = torch.split(positive_rois, 4, dim=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = torch.split(roi_gt_boxes, 4, dim=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = torch.cat([y1, x1, y2, x2], 1)
        box_ids = torch.range(0, roi_masks.size(0))
        # 从roi_masks中切出boxes，再resize到config.MASK_SHAPE大小
        # masks = torch.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
        #                                     box_ids,
        #                                     config.MASK_SHAPE)
        crfuc = CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)
        masks = Variable(crfuc(roi_masks.unsqueeze(1), boxes, box_ids).data, requires_grad=False)

        # Remove the extra dimension from masks.
        # transposed的时候增加了一维度 expand_dims(*,*,*,-1)
        masks = torch.squeeze(masks, dim=3)
        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss. 应用二值交叉熵损失前，进行round处理
        masks = torch.round(masks)

        # 7、组合正负ROIs，并进行零填充

        # Append negative ROIs and pad bbox deltas and masks that
        # are not used for negative ROIs with zeros.
        rois = torch.cat([positive_rois, negative_rois], dim=0)
        N = negative_rois.size(0)
        P = torch.max(config.TRAIN_ROIS_PER_IMAGE - rois.size(0), 0)
        rois = torch.pad(rois, [(0, P), (0, 0)])
        roi_gt_boxes = torch.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
        roi_gt_class_ids = torch.pad(roi_gt_class_ids, [(0, N + P)])
        deltas = torch.pad(deltas, [(0, N + P), (0, 0)])
        masks = torch.pad(masks, [[0, N + P], (0, 0), (0, 0)])

        return rois, roi_gt_class_ids, deltas, masks


class RoiTransformLayer(nn.Module):
    def __init__(self):
        super(RoiTransformLayer, self).__init__()

    def forward(self, *inputs):
        pass
        return inputs
