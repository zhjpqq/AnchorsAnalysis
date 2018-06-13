#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/6/11 16:19'
__author__ = 'ooo'

import torch
import numpy as np


@staticmethod
def detection_targets_graph_nocrowd(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """
    deperated deperated deperated !!!!!
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

    proposals = proposals.data
    gt_class_ids, gt_boxes, gt_masks = gt_class_ids.data, gt_boxes.data, gt_masks.data

    # 1. 移除proposals，GT中的0填充

    # 2. 区分拥挤GT & 不拥挤GT
    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    # 当一个BOX围住好几个物体实例时，称为Crowdbox，将其从训练阶段排除，给予一个负标签
    crowd_ix = torch.from_numpy(np.where(gt_class_ids < 0)[0]).long()
    non_crowd_ix = torch.from_numpy(np.where(gt_class_ids > 0)[0]).long()

    crowd_gt_class_ids = torch.gather(gt_class_ids, index=crowd_ix)
    crowd_gt_boxes = torch.gather(gt_boxes, dim=0, index=crowd_ix.unsqueeze(-1).expand(crowd_ix.size(0), 4))
    crowd_ix_mask = crowd_ix.unsqueeze(0).unsqueeze(0).expand(gt_masks.shape[0:2] + (crowd_ix.size(0),))
    crowd_gt_masks = torch.gather(gt_masks, dim=2, index=crowd_ix_mask)

    gt_class_ids = torch.gather(gt_class_ids, index=non_crowd_ix)
    gt_boxes = torch.gather(gt_boxes, index=non_crowd_ix.unsqueeze(-1).expand(non_crowd_ix.size(0), 4))
    no_crowd_ix_mask = non_crowd_ix.unsqueeze(0).unsqueeze(0).expand(gt_masks.shape[0:2] + (non_crowd_ix.size(0),))
    gt_masks = torch.gather(gt_masks, dim=2, index=no_crowd_ix_mask)

    # 3、计算proposals和gt_boxes的Overlaps

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = utils.bbox_overlaps(proposals, gt_boxes)  # shape: N×K
    crowd_overlaps = utils.bbox_overlaps(proposals, crowd_gt_boxes)
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
    # Permute masks to [N, channel, height, width] 跟box的坐标相对应
    transposed_masks = torch.unsqueeze(gt_masks.permute(2, 0, 1), 1)
    # Pick the right mask for each ROI
    roi_masks_index = roi_gt_box_assignment.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    roi_masks_index = roi_masks_index.expand((roi_gt_box_assignment.size(0),) + transposed_masks.shape[1:])
    roi_masks = torch.gather(transposed_masks, dim=0, index=roi_masks_index)
    # Compute mask targets, 挖出与roi相对应的mask
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI corrdinates from normalized image space to normalized mini-mask space.
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
    masks = Variable(crfuc(roi_masks, boxes, box_ids).data, requires_grad=False)

    # Remove the extra channel dimension from masks.
    masks = torch.squeeze(masks, dim=1)
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
