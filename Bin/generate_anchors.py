#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/6/18 15:05'
__author__ = 'ooo'

import os
import sys
import math
import time
import cv2
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from DataSets.coco_dataset import CocoDataset
from Configs.config import CocoConfig
from Models.backbone import backbone
from Models.fusionnet import fusionnet
from Layers.centers import generate_centers, compute_distance, compute_matches, compute_histogram

# 设定路径
curr_dir = os.getcwd()
root_dir = os.path.dirname(curr_dir)
exp_dir = os.path.join(root_dir, 'Experiments', 'anchors')
assert os.path.exists(exp_dir)

backbone_dir = ['/data/zhangjp/HRCNN/Backbones', '/data/zhangjp/HotAnchorRCNN_PyTorch/Backbones'][0]
backbone_name = ['resnet50-19c8e357.pth', 'resnet101-5d3b4d8f.pth'][0]
assert os.access(os.path.join(backbone_dir, backbone_name), os.R_OK)

data_dir = '/data/dataset/MSCOCO/data'
data_year = '2014'

# 配置超参数
config = CocoConfig()
config.EXP_DIR = exp_dir
config.DATASET = ['train', 'val', 'minival', 'valminusminival', ][2]

config.BACKBONE_ARCH = 'resnet50'
config.BACKBONE_DIR = backbone_dir
config.BACKBONE_NAME = backbone_name
config.BACKBONE_STAGES = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'][0:3]

config.ANCHOR_STAGE = 2
config.ANCHOR_METHOD = ['uniform', 'edges', 'hot', 'fpn'][2]
config.ANCHORS_PER_IMAGE = 50000

# 构造coco数据集
if config.DATASET == 'train':
    dataset_train = CocoDataset(config=config)
    dataset_train.load_coco(data_dir=data_dir,
                            subset='train',
                            year=data_year,
                            return_coco=True,
                            auto_download=False)
    dataset_train.load_coco(data_dir=data_dir,
                            subset='valminusminival',
                            year=data_year,
                            return_coco=True,
                            a10000uto_download=False)
    dataset_train.prepare()
    dataset = dataset_train
elif config.DATASET == 'minival':
    dataset_val = CocoDataset(config=config)
    dataset_val.load_coco(data_dir=data_dir,
                          subset='minival',
                          year=data_year,
                          return_coco=True,
                          auto_download=False)
    dataset_val.prepare()
    dataset = dataset_val
else:
    dataset = None

# data iterator # not generator!
# trainset_iter = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=4)
# valset_iter = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=4)

dataset_iter = dataset.generator_box(config, batch_size=1, shuffle=True, augment=False, format='Tensor',
                                     data_label=False)

# 构造网络模型
model = backbone(arch=config.BACKBONE_ARCH, pretrained=True, model_dir=backbone_dir, model_name=backbone_name)
fusion = fusionnet(method=config.FUSION_METHOD,
                   levels=config.FUSION_LEVELS,
                   indepth=config.FUSION_CHANNELS_IN,
                   outdepth=config.FUSION_CHANNELS_OUT,
                   strides=config.FUSION_STRIDES,
                   shapes=config.FUSION_SHAPES,
                   stages=config.BACKBONE_STAGES)
model.eval()
fusion.eval()
model.cuda()
# 采样命中率PDF
best_match_dist = []

# 采样效率PDF
hitok_rate = []
gtbox_nums = []
anchor_nums = config.ANCHORS_PER_IMAGE

#
image_nums = dataset.image_nums
idx = 0
stop_idx = image_nums

start = time.time()
for inputs in dataset_iter:
    #  Wrap all Tensor in Variable
    images = Variable(inputs[0]).cuda()
    image_metas = inputs[1][0]
    gt_class_ids = Variable(inputs[2][0]).cuda()
    gt_boxes = Variable(inputs[3][0]).cuda()  # [y, x, h, w]

    if 0:
        cv2.cvtColor()
        image = images[0].data.permute(1, 2, 0).cpu().numpy()
        image = CocoDataset.unmold_image(image, config.MEAN_PIXEL)
        cv2.imshow('images', image)

    # [C0, C1, C2, C3, C4, C5, C6]
    feature_maps = model(images, stages=config.BACKBONE_STAGES)

    if 0:
        image = feature_maps[0][0].data.permute(1, 2, 0).cpu().numpy()
        image = CocoDataset.unmold_image(image, config.MEAN_PIXEL)
        print('image.shape', image.shape)
        cv2.imshow('images', image)

    # [P2, P3, P4, P5, P6]
    # feature_maps = fusion(feature_maps[1:])

    centers = generate_centers(feature_maps, config, image_meta=image_metas, method=config.ANCHOR_METHOD)
    feature_maps, images = None, None

    # [N, K, (dist)]
    distance = compute_distance(centers, gt_boxes)

    # [d0, d1, d2, ...] or hitok
    # matches = compute_matches(distance, threshold=None)

    # 计算命中率PDF
    matches = compute_matches(distance, threshold=None, verbose=0)
    best_match_dist.extend(matches)

    # 计算采样效率曲线
    hitok, gtbox = compute_matches(distance, threshold=5, verbose=0)
    hitok_rate.append(hitok)
    gtbox_nums.append(gtbox)

    if idx < stop_idx:
        idx += 1
        print('next batch, idx is %s/%s, %0.1f%% done.' % (idx, stop_idx, idx*100 / stop_idx))
    else:
        break

timeit = np.round((time.time() - start) / 60, decimals=4)
print('喵，喵，喵，喵 .... %s 分 ' % timeit)

best_match_dist = np.array(best_match_dist, dtype=np.float32)
hitok_rate = np.array(hitok_rate, dtype=np.float32)
gtbox_nums = np.array(gtbox_nums, dtype=np.float32)
hitok_xiaolv = hitok_rate / anchor_nums

gtbox_total = np.sum(gtbox_nums)
images_total = hitok_rate.shape[0]

hitok_mean = np.mean(hitok_rate)
best_match_mean = np.mean(best_match_dist)

print('total images: %s,  total gtbox: %s, best-match-dist-mean: %s, hitok-rate-mean: %s'
      % (images_total, gtbox_total, best_match_mean, hitok_mean))


fig = plt.figure()
ax11 = fig.add_subplot(2, 2, 1)
ax12 = fig.add_subplot(2, 2, 2)
ax21 = fig.add_subplot(2, 2, 3)
ax22 = fig.add_subplot(2, 2, 4)

ax11.hist(best_match_dist, bins=200, range=(0, 60), normed=False)
ax12.hist(best_match_dist, bins=200, range=(0, 60), normed=True)
ax11.set_title('data: %s, method: %s, nums: %s, dist-mean: %s' % (
               'val', config.ANCHOR_METHOD, config.ANCHORS_PER_IMAGE, best_match_mean))

ax21.hist(hitok_rate, bins=20, normed=False)  # (1-0)/20 = 0.05
ax22.hist(gtbox_nums, bins=20, normed=True)
ax21.set_title('hitok rate mean : %s' % hitok_mean)

plt.show()

fname = '%s-hitok-%s-%s.png' % (config.ANCHOR_METHOD, config.DATASET, config.ANCHORS_PER_IMAGE)
filepath = os.path.join(config.EXP_DIR, fname)
print('file saved at: ', filepath)
fig.savefig(filename=filepath, format='png', transparent=False, dpi=300, pad_inches=0)
