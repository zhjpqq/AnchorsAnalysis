#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/4/6 23:49'
__author__ = 'ooo'

import numpy as np
import torch
from torch.autograd import Variable

from DataSets.imdb import IMDB
from NMS1.nms.nms_wrapper import nms as nms_func1
from NMS2.nms.nms_wrapper import nms as nms_func2
from Utils import utils


############################################################
#  Detection Layer
############################################################


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= torch.exp(deltas[:, 2])
    width *= torch.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result


def clip_to_window(window, boxes):
    """
        window: (y1, x1, y2, x2). The window in the image we want to clip to.
        boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, 0] = boxes[:, 0].clamp(float(window[0]), float(window[2]))
    boxes[:, 1] = boxes[:, 1].clamp(float(window[1]), float(window[3]))
    boxes[:, 2] = boxes[:, 2].clamp(float(window[0]), float(window[2]))
    boxes[:, 3] = boxes[:, 3].clamp(float(window[1]), float(window[3]))

    return boxes


def refine_detections(rois, probs, deltas, window, config):
    """Refine classified rois and filter overlaps and return final detections.
    精调分类后的rois，过滤重叠部分，返回最后的检出.

    冗余class过滤，0面积过滤，背景过滤，低分过滤，同类抑制，数量过滤

    1.  按分类scores过滤掉冗余class, 得到(N, class_id, <score, delta>)
        先按score筛选出正确的class, 再查找此class对应的delta:
        (N, num_classes, score) & (N, num_classes, (dy, dx, log(dh), log(dw))) .

    2.  将delta应用到box，并裁剪box到window范围, 并过滤掉面积为零的box.

    3.  过滤掉背景box，class_id<0.

    3.  过滤掉分类得分较低的box, config.DETECTION_MIN_CONFIDENCE.

    4.  对每个类的所有检出box做非极大值抑制, NMS, config.DETECTION_NMS_THRESHOLD.

    5.  根据检出数量de限制过滤box, Top, config.DETECTION_MAX_INSTANCES

    https://github.com/zhjpqq/pytorch-mask-rcnn/blob/master/model.py   torch
    https://github.com/zhjpqq/Mask_RCNN/blob/master/model.py           numpy

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, (score1, score2, score3, ...)]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """

    # Class IDs per ROI
    _, class_ids = torch.max(probs, dim=1)  # shape: N

    # Class probability of the top class of each ROI
    # Class-specific bounding box deltas
    idx = torch.arange(class_ids.size(0)).long().cuda()  # shape: N
    class_scores = probs[idx, class_ids.data]            # shape: N
    deltas_specific = deltas[idx, class_ids.data]        # shape: N×4 索引可以不写全[idx, class_ids.data, :]

    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    std_dev = Variable(torch.FloatTensor(np.reshape(config.BBOX_STD_DEV, [1, 4])).cuda(), requires_grad=False)
    refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)    # shape: N×4

    # Convert coordiates to image domain
    height, width = config.IMAGE_SHAPE[:2]
    scale = Variable(torch.FloatTensor(np.array([height, width, height, width])).cuda(), requires_grad=False)
    refined_rois *= scale
    # Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)

    # Round and cast to int since we're deadling with pixels now
    refined_rois = torch.round(refined_rois)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep_bool = class_ids > 0       # shape: N

    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        keep_bool &= (class_scores >= config.DETECTION_MIN_CONFIDENCE)   # shape: N  todo ??? 0*
    keep = torch.nonzero(keep_bool)[:, 0]     # shape: N

    # Apply per-class NMS1
    pre_nms_class_ids = class_ids[keep.data]  # shape: N
    pre_nms_scores = class_scores[keep.data]  # shape: N
    pre_nms_rois = refined_rois[keep.data]    # shape: Nx4

    for i, class_id in enumerate(utils.unique1d(pre_nms_class_ids)):
        # Pick detections of this class
        ixs = torch.nonzero(pre_nms_class_ids == class_id)[:, 0]

        # Sort
        ix_rois = pre_nms_rois[ixs.data]        # shape: Nx4
        ix_scores = pre_nms_scores[ixs.data]    # shape: N
        ix_scores, order = ix_scores.sort(descending=True)  # shape: N
        ix_rois = ix_rois[order.data, :]

        class_keep = nms_func1(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).data, config.DETECTION_NMS_THRESHOLD)
        # shape: nx1 torch.cuda.IntTensor

        # Map indicies
        # a = class_keep[:, 0].long(), b = order[a], c = ixs[b], d = keep[c]

        # original: class_keep = keep[ixs[order[class_keep].data].data]
        class_keep = keep[ixs[order[class_keep[:, 0].long()]]]   # shape: n

        if i == 0:
            nms_keep = class_keep
        else:
            nms_keep = utils.unique1d(torch.cat((nms_keep, class_keep)))
    keep = utils.intersect1d(keep, nms_keep)

    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids = class_scores[keep.data].sort(descending=True)[1][:roi_count]
    keep = keep[top_ids.data]

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    result = torch.cat((refined_rois[keep.data],
                        class_ids[keep.data].unsqueeze(1).float(),
                        class_scores[keep.data].unsqueeze(1)), dim=1)

    return result   # shape: nx6, 100x6


def detection_layer(config, rois, class_probs, bbox_deltas, image_metas, normalized=False):
    """Takes classified proposal boxes and their sores & bounding box deltas and
    returns the final detection boxes.

    输入：
    rois : [batch, N, (y1, x1, y2, x2)], tensor
    class_probs : [batch, N, (score1，score2, score3, ...)], tensor      # batch*N ?
    bbox_deltas : [batch, N, class_nums, (dy1, dx1, dh, dw)], tensor    # batch*N ?
    image_metas : [batch, (meta)], tensor                               # batch*N ?

    Returns:
    all_detections: [batch, max_N, (y1, x1, y2, x2, class_id, score)]  in pixels
    all_boxes : [batch, max_N, (y1, x1, y2, x2)]  in normalized
    """

    # Currently only supports batchsize 1
    rois = rois.squeeze(0)
    _, _, window, _ = IMDB.parse_image_meta(image_metas)
    window = window[0]

    all_detections = refine_detections(rois, class_probs, bbox_deltas, window, config)

    all_boxes = all_detections[:, :4]
    if normalized:
        h, w = config.IMAGE_SHAPE[:2]
        scale = Variable(torch.FloatTensor(np.array([h, w, h, w])).cuda(), requires_grad=False)
        all_boxes /= scale

    all_detections = all_detections.unsqueeze(0)
    all_boxes = all_boxes.unsqueeze(0)

    return all_detections, all_boxes

    # # Goto surpport batchsize > 1
    # all_detections = torch.zeros((rois.size(0), config.DETECTION_MAX_INSTANCES, 6)).cuda()
    # _, _, window, _ = IMDB.parse_image_meta(image_metas)
    #
    # for b in range(rois.size(0)):
    #     detections = refine_detections(rois[b], class_probs[b], bbox_deltas[b], window[b], config)
    #     all_detections[b, :, :] = detections
    #
    # all_boxes = all_detections[:, :, :4]
    # if normalized:
    #     h, w = config.IMAGE_SHAPE[:2]
    #     scale = Variable(torch.FloatTensor(np.array([h, w, h, w])).cuda(), requires_grad=False)
    #     all_boxes /= scale
    #
    # return all_detections, all_boxes


def pyramid_detection_layer(config, rois, class_probs, bbox_deltas, image_metas, normalized=False):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    输入：
    rois : [batch, N, (y1, x1, y2, x2)], tensor
    class_probs : [batch, N, (score1，score2, score3, ...)], tensor    # batch*N ?
    bbox_deltas : [batch, N, class_nums, (dy1, dx1, dh, dw)], tensor  # batch*N ?
    image_metas : [batch, (meta)], tensor                             # batch*N ?

    [rois1, rois2, rois3, ...],
    [class_probs1, class_probs2, class_probs3, ...],
    [bbox_deltas, bbox_deltas, bbox_deltas, ...],
    [image_metas, image_metas, image_metas, ...],

    Returns:
    all_detections: [[batch, max_N, (y1, x1, y2, x2, class_id, score)], ...]  in pixels
    all_boxes : [[batch, max_N, (y1, x1, y2, x2)], ...]  in normalized
    """

    if config.FEATURE_FUSION_LEVELS == 1:

        all_detections, all_boxes = detection_layer(config, rois, class_probs, bbox_deltas, image_metas, normalized)

        return all_detections, all_boxes

    elif config.FEATURE_FUSION_LEVELS > 1:

        assert not np.any(np.array([len(class_probs), len(bbox_deltas), len(image_metas)]) - len(rois))

        all_detections = []
        all_boxes = []
        for level in range(len(rois)):
            detections, boxes = detection_layer(config, rois[level], class_probs[level],
                                                bbox_deltas[level], image_metas[level], normalized)
            all_detections.append(detections)
            all_boxes.append(boxes)

        return all_detections, all_boxes

    else:
        raise ValueError('错误的特征融合级数！')