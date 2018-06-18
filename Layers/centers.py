#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/6/18 16:13'
__author__ = 'ooo'

import numpy as np
import math
import os
import cv2
import torch
from torch.autograd import Variable


def generate_centers(fmaps, counts, imag_shape, method='uniform'):
    if method == 'uniform':
        return uniform_centers(fmaps, counts, imag_shape)

    elif method == 'edges':
        return edges_centers(fmaps, counts, imag_shape)

    elif method == 'hot':
        return hot_centers(fmaps, counts, imag_shape)

    elif method == 'fpn':
        return fpn_centers(fmaps, counts, imag_shape)


def uniform_centers(fmaps, counts, imag_shape, anchor_stride=1):
    # fmaps [b, c, h, w]
    fmaps = fmaps.data[0, :, :, :]
    h, w = fmaps.shape[1:]
    fmap_stride = imag_shape / h
    y = np.arange(0, h, anchor_stride) * fmap_stride
    x = np.arange(0, h, anchor_stride) * fmap_stride
    y, x = np.meshgrid(y, x)
    centers = np.stack([y.reshape(-1), x.reshape(-1)], axis=1)  # N x 2
    if centers.shape[0] > counts:
        index = np.arange(centers.shape[0])
        index = np.random.choice(index, counts, replace=False)
        centers = centers[index, :]
    else:
        raise Exception('锚点数量不足, %s' % centers.shape[0])
    return centers


def edges_centers(fmaps, counts, imag_shape):
    # fmaps [b, c, h, w]
    fmaps = fmaps.data[0, :, :, :]
    h, w = fmaps.shape[1:]
    fmap_stride = imag_shape / h
    


def hot_centers(fmaps, counts, imag_shape):
    pass


def fpn_centers(fmaps, counts, imag_shape):
    pass


def compute_distance(centers, boxes, format='ndarry'):
    # centers [N, (y, x)]
    # boxes [K, (y, x, h, w)]
    boxes = boxes[:, 0: 2]

    if isinstance(centers, Variable):
        centers = centers.data
    if isinstance(boxes, Variable):
        boxes = boxes.data
    if isinstance(centers, (torch.FloatTensor, torch.IntTensor)):
        centers = centers.cpu().numpy()
    if isinstance(boxes, (torch.FloatTensor, torch.IntTensor)):
        boxes = boxes.cpu().numpy()

    N, K = centers.shape[0], boxes.shape[0]

    centers = np.concatenate([centers] * K, axis=0)
    boxes = np.repeat(boxes, N, axis=0)
    delta = centers - boxes
    distance = np.sqrt(delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1])
    distance = distance.reshape([N, K])

    if format == 'Tensor':
        distance = torch.from_numpy(distance).float()
    elif format == 'Variable':
        distance = Variable(torch.from_numpy(distance).float(), requires_grad=False)
    else:
        distance = distance
    return distance


def compute_matches(distance, threshold=None):
    # distance [N, K, (dist)]
    # N个采样得到的点， K个gt-box的中心

    if isinstance(distance, Variable):
        distance = distance.data
    if isinstance(distance, (torch.FloatTensor, torch.IntTensor)):
        distance = distance.cpu().numpy()

    N, K = distance.shape

    # 返回K个最佳(最小)匹配值，用于后续‘落袋计数’统计直方图
    if threshold is None:
        matches = np.min(distance, axis=0)
        return matches

    # 返回当前图片上的命中率 K'/K
    else:
        matches = (distance <= threshold).astype(np.int)
        matches = np.sum(np.max(matches, axis=0))
        hitok = matches / K
        return hitok


def compute_histogram(hist, arr=None, normalized=False):
    """
    将arr中的值，统计到hist中, 再可选归一化.
    arr = [a1,a2,a3,...]
    hist = [N_BINS, Counts], N_BINS: 1, 2, 3, 6, 7, 16, ..., 升序
    """
    assert type(hist) == np.ndarray and hist.ndim == 2
    assert type(arr) == np.ndarray and arr.ndim == 1
    assert np.all(np.sort(hist[:, 0]) == hist[:, 0]), 'N_BINS should in assend'

    hist = hist.astype(np.float)

    if arr is not None:
        min, max = hist[0, 0], hist[-1, 0]
        for x in arr:
            x = np.round(x)
            if x < min:
                hist[0, 1] += 1
            elif x > max:
                hist[-1, 1] += 1
            else:
                index = np.where(hist[:, 0] >= x)[0][0]
                hist[index, 1] += 1
    if normalized:
        hist[:, 1] = hist[:, 1] / np.sum(hist[:, 1])

    return hist
