#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/6/18 15:05'
__author__ = 'ooo'

import os
import sys
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from DataSets.coco_dataset import CocoDataset
from Configs.config import CocoConfig
from Models.backbone import backbone
from Models.fusionnet import fusionnet
from Layers.centers import generate_centers, compute_distance, compute_matches

# 设定路径
curr_dir = os.getcwd()
root_dir = os.path.dirname(curr_dir)
exp_dir = os.path.join(root_dir, 'Experiments', 'coco_exp')

backbone_dir = os.path.join(root_dir, 'Backbones')
backbone_name = 'resnet50-19c8e357.pth'

data_dir = '/data/dataset/MSCOCO/data'
data_year = '2014'

# 配置超参数
config = CocoConfig()
config.EXP_DIR = exp_dir
config.BACKBONE_ARCH = 'resnet50'
config.BACKBONE_DIR = backbone_dir
config.BACKBONE_NAME = backbone_name

# 构造coco数据集
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
                        auto_download=False)
dataset_train.prepare()

dataset_val = CocoDataset(config=config)
dataset_val.load_coco(data_dir=data_dir,
                      subset='minival',
                      year=data_year,
                      return_coco=True,
                      auto_download=False)
dataset_val.prepare()

# data iterator # not generator!
trainset_iter = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=4)
valset_iter = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=4)

# 构造网络模型
model = backbone(arch='resnet50', pretrained=True, model_dir=backbone_dir, model_name=backbone_name)
fusion = fusionnet(stages=config.BACKBONE_STAGES,
                   method=config.FUSION_METHOD,
                   levels=config.FUSION_LEVELS,
                   indepth=config.FUSION_CHANNELS_IN,
                   outdepth=config.FUSION_CHANNELS_OUT,
                   strides=config.FUSION_STRIDES,
                   shapes=config.FUSION_SHAPES)
# 生成锚点
dataset_iter = [trainset_iter, valset_iter][0]
counts = 500
for inputs in dataset_iter:
    #  Wrap all Tensor in Variable
    images = Variable(inputs[0]).cuda()
    image_metas = inputs[1].numpy()
    gt_class_ids = Variable(inputs[2]).cuda()
    gt_boxes = Variable(inputs[3]).cuda()       # [y1, x1, y2, x2]

    model.eval()
    # [C0, C1, C2, C3, C4, C5, C6]
    feature_maps = model(images)

    # [P2, P3, P4, P5, P6]
    feature_maps = fusion(feature_maps[1:])

    centers = generate_centers(feature_maps, counts, method='uniform')

    distance = compute_distance(centers, gt_boxes)

    matches = compute_matches(distance, threshold=1.414)

    # 计算命中率PDF


    # 计算采样效率曲线
