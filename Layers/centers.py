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


def generate_centers(fmaps, counts, image_meta, method='uniform'):
    if method == 'uniform':
        fmap = fmaps[3]
        return uniform_centers(fmap, counts, image_meta)

    elif method == 'edges':
        fmap = fmaps[0]
        return edges_centers(fmaps, counts, image_meta)

    elif method == 'hot':
        fmap = fmaps[3]
        return hot_centers(fmap, counts, image_meta)

    elif method == 'fpn':
        return fpn_centers(fmaps, counts, image_meta)


def uniform_centers(fmap, counts, image_meta, anchor_stride=1):
    # fmap [b, c, h, w]
    fmap = fmap.data[0, :, :, :]
    h, w = fmap.shape[1:]
    fmap_stride = image_meta[1:3] / np.array([h, w])
    y = np.arange(0, h, anchor_stride) * fmap_stride
    x = np.arange(0, w, anchor_stride) * fmap_stride
    y, x = np.meshgrid(y, x)
    centers = np.stack([y.reshape(-1), x.reshape(-1)], axis=1)  # N x 2
    if centers.shape[0] >= counts:
        index = np.arange(centers.shape[0])
        index = np.random.choice(index, counts, replace=False)
        centers = centers[index, :]
    else:
        raise Exception('锚点数量不足, %s' % centers.shape[0])
    return centers


def edges_centers(fmap, counts, image_meta):
    # fmap [b, 3, h, w]
    assert fmap.shape[1] == 3
    fmap = fmap.data[0, :, :, :].permute(1, 2, 0).cpu().numpy()
    fstride = image_meta[1:3] / np.array(fmap.shape[0:2])
    window = image_meta[4:8]  # (y1, x1, y2, x2) in image cooredinates

    fmap = cv2.cvtColor(fmap, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(fmap, 160, 200)
    # cv2.imshow("fmap", fmap)
    # cv2.imshow("edges", fmap)
    rows, cols = np.where(edges)
    rows_in_win = np.logical_and(rows > window[0], rows < window[2])
    cols_in_win = np.logical_and(cols > window[1], cols < window[3])
    ctrs_in_win = np.logical_and(rows_in_win, cols_in_win)
    centers = np.stack([rows, cols], axis=1)
    centers = centers[ctrs_in_win, :]
    centers *= fstride
    if centers.shape[0] >= counts:
        index = np.arange(centers.shape[0])
        index = np.random.choice(index, counts, replace=False)
        centers = centers[index, :]
    else:
        raise Exception('锚点数量不足, %s' % centers.shape[0])
    return centers


def hot_centers(fmap, counts, image_meta):
    """方案3
    累积最大热度：将所有通道独立去均值，独立求绝对值，再映射到01？，再累积相加，然后比大小.
    返回的centers，必须是以原图尺寸为参照！
    :param  fmap: [batch, channels, height, weight]，float32
    :param  shape: (h, w, c) 原图尺寸
    :return: centers: [N, (y, x)] tensor, Normalized in image_shape.
    """
    assert fmap.size(0) == 1, 'batch size should be 1'
    assert fmap.size(2) * fmap.size(3) >= counts, 'need anchors counts >= pixels of fmap :%s & %s'
    fmap = fmap.data[0].cpu().numpy()
    image_shape = image_meta[1:4]
    window = image_meta[4:8]
    fstride = image_shape / np.array(fmap.shape[2:])
    fmap = np.sum(np.abs(fmap - np.mean(np.mean(fmap, axis=1, keepdims=True), axis=2, keepdims=True)), axis=0)
    # fmap = np.sum(np.abs(fmap), axis=0)
    kind = np.argsort(fmap.reshape(-1), axis=0)
    kval = fmap.reshape(-1)[kind[counts]]
    rows, cols = np.where(fmap >= kval)
    rows_in_win = np.logical_and(rows > window[0], rows < window[2])
    cols_in_win = np.logical_and(cols > window[1], cols < window[3])
    ctrs_in_win = np.logical_and(rows_in_win, cols_in_win)
    centers = np.stack([rows, cols], axis=1)
    centers = centers[ctrs_in_win, :]
    centers *= fstride
    if centers.shape[0] >= counts:
        index = np.arange(centers.shape[0])
        index = np.random.choice(index, counts, replace=False)
        centers = centers[index, :]
    else:
        raise Exception('锚点数量不足, %s' % centers.shape[0])
    return centers


def fpn_centers(fmaps, counts, image_meta):
    pass


def compute_distance(centers, boxes, format='ndarry'):
    # centers [N, (y, x)]
    # boxes [K, (y, x, h, w)]
    # distance [N, K, (dist)]
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
    # matches [d0,d1,d2,...] or [hitok0, hitok1, hitok2, ...]

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

    hist = hist.astype(np.float32)
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
