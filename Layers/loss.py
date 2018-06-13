#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/4/7 14:16'
__author__ = 'ooo'

import torch
import torch.nn.functional as F
from torch.autograd import Variable


############################################################
#  Loss Functions
############################################################

# 标注方法：
# [b, N, (class_id)] & [b, N, (class_id，)] 分别表示 shape: b*N, b*N*1。逗号元组以示独占一个维度。
# [b, N, (dy, dx, dw, dh)] shape: b*N*4

# target_class_ids: [b, N, (class_id)]      class_logits: [b*N, class_nums, (logits)]
# target_deltas: [b, N, (dy, dx, dw, dh)]   pred_deltas: [b*N, class_nums, (dy, dx, dw, dh)]
# target_masks: [b, N, h, w]                pred_masks: [b*N, class_nums, h', w']
def compute_losses(target_class_ids, class_logits, target_deltas, pred_deltas, target_masks, pred_masks):
    target_class_ids = target_class_ids.view(-1)   # shape [b, N] -> [b*N]
    target_deltas = target_deltas.view(-1, target_deltas.size(-1))
    target_masks = target_masks.view(-1, target_masks.size(-2), target_masks.size(-1))

    class_loss = compute_class_loss(target_class_ids, class_logits)
    bbox_loss = compute_bbox_loss(target_deltas, pred_deltas, target_class_ids)
    mask_loss = compute_mask_loss(target_masks, pred_masks, target_class_ids)

    return [class_loss, bbox_loss, mask_loss]


def compute_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.
    # todo # https://github.com/zhjpqq/pytorch-mask-rcnn/blob/master/model.py#L1047 ??? 参数不匹配
    target_class_ids: [batch*num_rois]. Integer class IDs. Uses zero padding to fill in the array.
    pred_class_logits: [batch*num_rois, num_classes]
    """
    # Loss
    if target_class_ids.size():
        loss = F.cross_entropy(pred_class_logits, target_class_ids.long())
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()
    return loss


def compute_bbox_loss(target_bbox, pred_bbox, target_class_ids):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch*num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch*num_rois]. Integer class IDs.
    pred_bbox: [batch*num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    if target_class_ids.size():
        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.data].long()
        indices = torch.stack((positive_roi_ix, positive_roi_class_ids), dim=1)

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = target_bbox[indices[:, 0].data, :]
        pred_bbox = pred_bbox[indices[:, 0].data, indices[:, 1].data, :]  # [batch_index, class_index]
        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()
    return loss


def compute_mask_loss(target_masks, pred_masks, target_class_ids):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch*num_rois, height, width].
                  A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch*num_rois]. Integer class IDs. Zero padded.
    pred_masks:  [batch*num_rois, class_nums, height, width] float32 tensor with values from 0 to 1.
    """
    if target_class_ids.size():
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[indices[:, 0].data, :, :]
        y_pred = pred_masks[indices[:, 0].data, indices[:, 1].data, :, :]  # [batch_index, class_index]

        # Binary cross entropy
        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss
