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
from Utils import utils
from DataSets.imdb import IMDB
from DataSets.coco_dataset import CocoDataset


def generate_centers(fmaps, config, image_meta, method='uniform'):
    if method == 'uniform':
        fmap = fmaps[config.ANCHOR_STAGE]
        return uniform_centers(fmap, config, image_meta)

    elif method == 'edges':
        fmap = fmaps[config.ANCHOR_STAGE]
        return edges_centers(fmap, config, image_meta)

    elif method == 'hot':
        fmap = fmaps[config.ANCHOR_STAGE]
        return hot_centers(fmap, config, image_meta)

    elif method == 'fpn':
        return fpn_centers(fmaps, config, image_meta)


def uniform_centers(fmap, config, image_meta, anchor_stride=7):
    # fmap [b, c, h, w]
    fmap = fmap.data[0, :, :, :]
    h, w = fmap.shape[1:]
    image_shape = config.IMAGE_SHAPE
    counts = config.ANCHORS_PER_IMAGE
    anchor_stride = np.floor(image_shape[0]/np.sqrt(counts))

    fmap_stride = image_shape[1] / h
    y = np.arange(0, h * fmap_stride, anchor_stride)
    x = np.arange(0, w * fmap_stride, anchor_stride)
    y, x = np.meshgrid(y, x)
    centers = np.stack([y.reshape(-1), x.reshape(-1)], axis=1)  # N x 2
    if centers.shape[0] >= counts:
        index = np.arange(centers.shape[0])
        index = np.random.choice(index, counts, replace=False)
        centers = centers[index, :]
    else:
        raise Exception('锚点数量不足, %s' % centers.shape[0])
    return centers


def edges_centers(fmap, config, image_meta):
    # fmap [b, 3, h, w]
    # centers: [N, (y, x)]
    assert fmap.shape[1] == 3
    fmap = fmap.data[0, :, :, :].permute(1, 2, 0).cpu().numpy()
    image_shape = config.IMAGE_SHAPE
    fstride = image_shape[0:2] / np.array(fmap.shape[0:2])
    window = image_meta[4:8].astype(np.int32)  # (y1, x1, y2, x2) in image cooredinates
    counts = config.ANCHORS_PER_IMAGE

    fmap = IMDB.unmold_image(fmap, rbg_mean=config.MEAN_PIXEL)
    fmap = cv2.cvtColor(fmap, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("fmap", fmap)
    # [100, 200] [60,200]/~2W [30 100]/3W-4W [10 80]/5W-6W [10 60]/7W-30W  [10 30]/50W
    edges = cv2.Canny(fmap, 10, 20)
    # cv2.imshow("edges", edges)
    rows, cols = np.where(edges)
    rows_in_win = np.logical_and(rows > window[0] + 1, rows < window[2] - 1)
    cols_in_win = np.logical_and(cols > window[1] + 1, cols < window[3] - 1)
    ctrs_in_win = np.logical_and(rows_in_win, cols_in_win)
    centers = np.stack([rows, cols], axis=1)
    centers = centers[ctrs_in_win, :]
    centers *= fstride.astype(np.int32)
    if centers.shape[0] >= counts:
        index = np.arange(centers.shape[0])
        index = np.random.choice(index, counts, replace=False)
        centers = centers[index, :]
    else:
        print('锚点数量不足---->>> url: %s >>> nums: %s' % (image_meta[0], centers.shape[0]))
        # raise Warning('锚点数量不足, %s' % centers.shape[0])
        if centers.shape[0] == 0:
            print('convert to uniform sampling ... ')
            y = np.arange(0, image_shape[0], 1)
            x = np.arange(0, image_shape[1], 1)
            y, x = np.meshgrid(y, x)
            centers = np.stack([y.reshape(-1), x.reshape(-1)], axis=1)  # N x 2
            index = np.arange(centers.shape[0])
            index = np.random.choice(index, counts, replace=False)
            centers = centers[index, :]
    return centers


def hot_centers(fmap, config, image_meta):
    """方案3
    累积最大热度：将所有通道独立去均值，独立求绝对值，再映射到01？，再累积相加，然后比大小.
    返回的centers，必须是以原图尺寸为参照！
    :param  fmap: [batch, channels, height, weight]，float32
    :param  shape: (h, w, c) 原图尺寸
    :return: centers: [N, (y, x)] tensor, Normalized in image_shape.
    """
    counts = config.ANCHORS_PER_IMAGE
    assert fmap.size(0) == 1, 'batch size should be 1'
    assert fmap.size(2) * fmap.size(3) >= counts, 'need anchors counts >= pixels of fmap :%s & %s'
    fmap = fmap.data[0].cpu().numpy()
    image_shape = config.IMAGE_SHAPE[0:2]
    fstride = image_shape / np.array(fmap.shape[1:])
    window = image_meta[4:8].astype(np.float32)  # y1 x1 y2 x2
    window /= np.concatenate((fstride, fstride), axis=0)
    window = np.round(window).astype(np.int32)

    # fmap = np.sum(np.abs(fmap - np.mean(np.mean(fmap, axis=1, keepdims=True), axis=2, keepdims=True)), axis=0)
    fmap = np.sum(np.abs(fmap), axis=0)

    rows = np.arange(fmap.shape[0])
    cols = np.arange(fmap.shape[1])
    # points 1000/[0.163 < 0.179 < 0.194]  1.5w[0.761<-<0.777]
    rows_out_win = np.logical_or(rows < window[0] + 2, rows > window[2] - 2)
    cols_out_win = np.logical_or(cols < window[1] + 2, cols > window[3] - 2)
    fmap[rows_out_win, :] = -99
    fmap[:, cols_out_win] = -99

    kval = np.sort(fmap, axis=None)[::-1][counts]
    if kval <= -99:
        kval = 0
        print('锚点数量不足---- <<< -99 >>> 0')
    rows, cols = np.where(fmap >= kval)
    centers = np.stack([rows, cols], axis=1)
    centers *= fstride.astype(np.int32)

    if centers.shape[0] >= counts:
        centers = centers[0:counts, :]
        # index = np.arange(centers.shape[0])
        # index = np.random.choice(index, counts, replace=False)
        # centers = centers[index, :]
    else:
        print('锚点数量不足---->>> url: %s >>> nums: %s' % (image_meta[0], centers.shape[0]))
        if centers.shape[0] == 0:
            print('convert to uniform sampling ... $%#%^^*&^*$%#%^^*&^*$%#%^^*&^*(&)$^&$^#$ ')
            y = np.arange(0, image_shape[0], 1)
            x = np.arange(0, image_shape[1], 1)
            y, x = np.meshgrid(y, x)
            centers = np.stack([y.reshape(-1), x.reshape(-1)], axis=1)  # N x 2
            index = np.arange(centers.shape[0])
            index = np.random.choice(index, counts, replace=False)
            centers = centers[index, :]
    return centers


def fpn_centers(fmaps, counts, image_meta):
    pass


def compute_distance(centers, boxes, format='ndarry'):
    # centers [N, (y, x)]
    # boxes [K, (y, x, h, w)]
    # distance [N, K, (dist)]
    boxes = utils.trim_zeros(boxes[:, 0: 2]).cpu()

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


def compute_matches(distance, threshold=None, verbose=0):
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
        if verbose:
            print('\n', matches, '\n')
        return matches

    # 返回当前图片上的命中率 K'/K, caiyangshu N
    else:
        matches = (distance <= threshold).astype(np.int)
        matches = np.sum(np.max(matches, axis=0))
        hitok = matches / K
        return hitok, K


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
