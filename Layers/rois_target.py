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


class RoiTargetLayer(nn.Module):
    def __init__(self, config):
        """
        数量限制：TRAIN_ROIS_PER_IMAGE < MAX_GT_INSTANCES < PROPOSALS_PER_IMAGE <  ANCHORS_PER_IMAGE
        :param config:
        """
        self.config = config
        super(RoiTargetLayer, self).__init__()

    def forward(self, proposals, gt_class_ids, gt_boxes, gt_masks):
        """
        输入：
        proposals: [[batch, N, (y1, x1, y2, x2)], ., ...], N=PROPOSALS_PER_IMAGE, 归一化，零填充. Var
        gt_class_ids: [batch, MAX_GT_INSTANCES] 类标. tensor
        gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] 归一化. tensor
        gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type. tensor
        输出：
        rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)], [rois1, rois2 ...] 归一化，零填充. tensor
        """
        # self.config.FUSION_LEVELS == 1 or > 1
        if len(proposals) == 1:

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

            return [rois], [target_class_ids], [target_deltas], [target_masks]

        # self.config.FUSION_LEVELS > 1:
        elif len(proposals) > 1:

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
        1。从proposal看，存在一个GT Iou >= 0.7 为正， 与所有GT的 IoU < 0.5为负， 介于0.5~0.7之间为中性.
        2。从GTbox角度看，与每个GT最近的那个proposal, 必须为正, 即每个GT都必须要有匹配对象.

        输入:
        proposals: [N, (y1, x1, y2, x2)] 归一化，零填充. Variable
        gt_class_ids: [MAX_GT_INSTANCES] 类标. Var
        gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] 未归一化，无零填充. Var
        gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type. Var

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

        # 0. 归一化gt_boxes坐标
        h, w = config.IMAGE_SHAPE[0:2]
        scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False).cuda()
        gt_boxes = gt_boxes / scale

        # 1. 移除proposals，GT中的0填充
        proposals = utils.trim_zeros(proposals)

        # 2. 区分拥挤GT & 不拥挤GT
        # Handle COCO crowds
        # A crowd box in COCO is a bounding box around several instances. Exclude
        # them from training. A crowd box is given a negative class ID.
        # 当一个BOX围住好几个物体实例时，称为Crowdbox，将其从训练阶段排除，给予一个负标签
        crowd_ix = np.where(gt_class_ids.data < 0)[0]
        if crowd_ix:
            crowd_ix = Variable(torch.from_numpy(crowd_ix).long().cuda(), requires_grad=False)
            crowd_gt_boxes = torch.gather(gt_boxes, dim=0, index=crowd_ix.unsqueeze(-1).expand(crowd_ix.size(0), 4))
        else:
            crowd_gt_boxes = Variable(torch.FloatTensor([])).cuda()
        # crowd_gt_class_ids = torch.gather(gt_class_ids, dim=0, index=crowd_ix)
        # crowd_gt_masks = torch.gather(gt_masks, dim=2, index=crowd_ix.unsqueeze(0).unsqueeze(0)
        #                                                     .expand(gt_masks.shape[0:2] + (crowd_ix.size(0),)))

        non_crowd_ix = Variable(torch.from_numpy(np.where(gt_class_ids.data > 0)[0]).long().cuda(), requires_grad=False)
        gt_class_ids = torch.gather(gt_class_ids, dim=0, index=non_crowd_ix)
        gt_boxes = torch.gather(gt_boxes, dim=0, index=non_crowd_ix.unsqueeze(-1).expand(non_crowd_ix.size(0), 4))
        gt_masks = torch.gather(gt_masks, dim=2, index=non_crowd_ix.unsqueeze(0).unsqueeze(0)
                                .expand(gt_masks.shape[0:2] + (non_crowd_ix.size(0),)))
        crowd_ix, non_crowd_ix = None, None

        # 3、计算proposals和gt_boxes的Overlaps
        # Compute overlaps matrix [proposals, gt_boxes]
        overlaps = utils.bbox_overlaps(proposals, gt_boxes)  # shape: N×K
        if crowd_gt_boxes:
            crowd_overlaps = utils.bbox_overlaps(proposals, crowd_gt_boxes)
            crowd_iou_max = torch.max(crowd_overlaps, dim=1)[0].data
            no_crowd_bool = (crowd_iou_max < 0.001)  # shape: N×K'
        else:
            no_crowd_bool = torch.ones(proposals.shape[0]).byte().cuda()
        crowd_overlaps, crowd_iou_max = None, None

        # 4、判定正负ROIs
        # Determine postive and negative ROIs

        # method1: 给定 counts & max & min threshold, firstly filter by max & min, then both randomly select P/N-rois
        # method2: 给定 counts & min, 首先选择 top-k(p_counts) as P-rois, others randomly selecte n_counts as N-rois
        # method1 可以始终确保正负ROI在一定阈值max&min的控制之内，但不能保证产生的数量，且需要手动设置这个max&min参数。
        # method2 可以始终确保前N个最大重叠度的ROI为正，而让负ROI随机产生。但是不能保证正负ROI一定符合max&min阈值要求，但也正好免去此手动参数。
        # method2 也可以只指定一个min threshold, 因为潜在负样本较多，所以不会产生实际负样本数量不足的现象。
        # method2 中直接在正样本中排除掉crowd gt box，不再考虑 no_crowd_bool 参数。
        # method3 是method1的另一种实现方式
        # 负样本应该尽可能为背景，不包含有效物体。
        # 负样本与非拥挤框的IOU<0.5, 且与拥挤框的IOU<0.001. 因为拥挤框内有多个完整物体，若某个proposal与之显著交叠，可能导致该物体成为负样本。

        method1, method2, method3 = (False, True, False)

        if method1:
            # dim1 : 每个proposal/roi的最佳gt_box的iou值
            roi_iou_max = torch.max(overlaps, dim=1)[0].data  # shape: N
            # 4.1. Positive ROIs are those with >= 0.5 IoU with a GT box
            positive_indices = np.where((roi_iou_max >= config.ROIS_GTBOX_IOU[0]))[0]  # shape: N
            positive_indices = torch.from_numpy(positive_indices).long().cuda()
            # 4.2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
            negative_indices = np.where(np.logical_and(roi_iou_max < config.ROIS_GTBOX_IOU[1], no_crowd_bool))[0]
            negative_indices = torch.from_numpy(negative_indices).long().cuda()

            # Subsample ROIs. Aim for 33% positive
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
            roi_iou_max = None

        elif method2:
            roi_iou_max = torch.max(overlaps, dim=1)[0].data  # shape: N
            roi_iou_max, roi_iou_ind = torch.sort(roi_iou_max, dim=0, descending=True)
            positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROIS_POSITIVE_RATIO)
            positive_indices = roi_iou_ind[0: positive_count]
            roi_iou_ind = roi_iou_ind[positive_count:]
            roi_iou_max = roi_iou_max[positive_count:]
            negative_count = config.TRAIN_ROIS_PER_IMAGE - positive_count
            negative_indices = roi_iou_ind[(roi_iou_max < config.ROIS_GTBOX_IOU[1]) & no_crowd_bool[roi_iou_ind]]
            print('positive_/negative_indices.numel()---@rois_target()--:', positive_indices.numel()/negative_indices.numel())
            vc = torch.randperm(negative_indices.numel())[0: negative_count].cuda()
            negative_indices = negative_indices[torch.randperm(negative_indices.numel())[0: negative_count].cuda()]
            roi_iou_max, roi_iou_ind, index = None, None, None

        elif method3:
            roi_iou_max = torch.max(overlaps, dim=1)[0].data  # shape: N
            roi_iou_ind = torch.arange(0, roi_iou_max.shape[0]).long().cuda()  # shape: N
            positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROIS_POSITIVE_RATIO)
            positive_indices = roi_iou_ind[roi_iou_max > config.ROIS_GTBOX_IOU[0]]
            positive_indices = positive_indices[torch.randperm(positive_indices.numel())[0: positive_count].cuda()]

            negative_count = config.TRAIN_ROIS_PER_IMAGE - positive_indices.shape[0]
            negative_indices = roi_iou_ind[(roi_iou_max < config.ROIS_GTBOX_IOU[1]) & no_crowd_bool[roi_iou_ind]]
            negative_indices = negative_indices[torch.randperm(negative_indices.numel())[0: negative_count].cuda()]
            roi_iou_max = None

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

        # Assign positive ROIs to GT boxes. 沿着gtbox方向(dim=1)统计最大值，得到每个roi的最佳gtbox/gt_class_id
        # index shape : N → N×1 → N×gt_boxes_count
        index = Variable(positive_indices.unsqueeze(-1).expand(positive_indices.size(0), overlaps.size(-1)),
                         requires_grad=False)
        positive_overlaps = torch.gather(overlaps, dim=0, index=index)  # N×K
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1].long()  # N
        # N → N×1 → N×4
        index = roi_gt_box_assignment.unsqueeze(-1).expand(roi_gt_box_assignment.size(0), gt_boxes.size(-1))
        roi_gt_boxes = torch.gather(gt_boxes, dim=0, index=index)
        roi_gt_class_ids = torch.gather(gt_class_ids, dim=0, index=roi_gt_box_assignment)
        gt_boxes, index = None, None

        # 6、计算ROIs的校正量deltas

        # Compute bbox refinement for positive ROIs # 对正ROIs计算bbox的修正量
        deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
        deltas /= Variable(torch.from_numpy(config.BBOX_STD_DEV).float().cuda(), requires_grad=False)

        # 6、抓取正ROI的-masks, 并计算mask targets

        # 在原始的GT-mask上，裁切位于roi_box中的那部分mask出来，再缩放到指定shape大小。
        # GT-mask是一个H×W的二值图，因此裁切出来的仍然是一个小二值图
        # 此小二值图，即是此roi_box的gt_mask_targets，可用于计算二值交叉熵损失
        # Assign positive ROIs to GT masks
        # Permute masks from [h, w, n] to [N, channel, height, width] 跟box的坐标相对应
        # transposed_masks = torch.transpose(torch.transpose(gt_masks, 2, 1), 1, 0).unsqueeze(1)
        transposed_masks = gt_masks.permute(2, 0, 1).unsqueeze(1)
        gt_masks = None

        # Pick the right mask for each ROI
        index = roi_gt_box_assignment.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # N -> Nx1x1x1
        index = index.expand((roi_gt_box_assignment.size(0),) + transposed_masks.shape[1:])  # Nx1x1x1 -> Nx1xHxW
        roi_masks = torch.gather(transposed_masks, dim=0, index=index)
        transposed_masks, roi_gt_box_assignment, index = None, None, None

        # Compute mask targets, 挖出与roi相对应的mask
        boxes = positive_rois
        if config.USE_MINI_MASK:
            # Transform ROI corrdinates from normalized image space to normalized mini-mask space.
            y1, x1, y2, x2 = torch.split(positive_rois, 1, dim=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = torch.split(roi_gt_boxes, 1, dim=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = torch.cat([y1, x1, y2, x2], 1)
        box_ids = Variable(torch.arange(0, roi_masks.size(0)), requires_grad=False).int().cuda()
        # 从roi_masks中切出boxes，再resize到config.MASK_SHAPE大小
        # crfuc(Nx1xHxW, Nx4, N) -> (N*1*28*28)
        # masks = torch.image.crop_and_resize(roi_masks.float(), boxes, box_ids, config.MASK_SHAPE)
        crfuc = CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)
        masks = Variable(crfuc(roi_masks, boxes, box_ids).data, requires_grad=False)
        roi_masks, box_ids = None, None

        # Remove the extra dimension from masks.
        masks = torch.squeeze(masks, dim=1)
        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss. 应用二值交叉熵损失前，进行round处理
        masks = torch.round(masks)

        # 7、组合正负ROIs，并进行零填充
        # Append negative ROIs and pad bbox deltas and masks that
        # are not used for negative ROIs with zeros.
        rois = torch.cat([positive_rois, negative_rois], dim=0)  # shape Nx4
        P = config.TRAIN_ROIS_PER_IMAGE - rois.size(0)
        P = P if P > 0 else 0
        N = negative_rois.size(0)
        rois = F.pad(rois, (0, 0, 0, P))
        # roi_gt_boxes = F.pad(roi_gt_boxes, (0, 0, 0, N + P))
        roi_gt_class_ids = F.pad(roi_gt_class_ids, (0, N + P))
        deltas = F.pad(deltas, (0, 0, 0, N + P))
        masks = F.pad(masks, (0, 0, 0, 0, 0, N + P))
        return rois, roi_gt_class_ids, deltas, masks
