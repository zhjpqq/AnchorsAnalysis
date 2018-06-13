#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/4/7 20:28'
__author__ = 'ooo'

import os
import time
import numpy as np
import zipfile
import urllib.request
import shutil

import torch
from torch import nn
from torch.nn import functional as F

from DataSets.shapes_dataset import ShapesDataset
from Configs.config import ShapesConfig
from Models.hot_rcnn import HotRCNN
from Utils import utils, visualize

#######################
#      路径配置
#######################
# Root directory of the project
curr_dir = os.getcwd()
root_dir = os.path.dirname(curr_dir)

# Experiments Log directory
exp_dir = os.path.join(root_dir, 'Experiments', 'shapes_exp')
log_dir = os.path.join(exp_dir, '**由函数self.set_log_dir()动态生成！**')

# pretrained models directory
backbone_dir = os.path.join(root_dir, 'Backbones')
backbone_name = 'resnet50-19c8e357.pth'
backbone_path = os.path.join(backbone_dir, backbone_name)
hotrcnn_path = None

#######################
#      模型训练
#######################

# Configurations
config = ShapesConfig()
config.EXP_DIR = exp_dir
config.BACKBONE_ARCH = 'resnet50'
config.BACKBONE_DIR = backbone_dir
config.BACKBONE_PATH = backbone_path
config.display()

# Training dataset
dataset_train = ShapesDataset(config)
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset(config)
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 2)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    # visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# Create model in training mode
model = HotRCNN(config=config)

# Which weights to start with ?
init_with = "ckpt"
# Weights 权值加载
# 1. 未训练加载：从imagenet加载resnet/resnext模型的权值
# 2. 已训完加载：加载一个已经训练完成的完整hrcnn模型
# 3. 半训练加载：从最近的训练节点加载一个半成品hrcnn模型
if init_with == 'backbone':  # 1
    model_path = model.get_backone_path()
elif init_with == 'hrcnn':  # 2
    model_path = model.get_weights_path('hrcnn', path='')
elif init_with == 'ckpt':  # 3
    model_path = model.get_ckpt_path()[1]
else:
    model_path = None
assert model_path is not None, 'No Model File is Found!'

print('开始加载预训练权值 ……', model_path)
model.load_weights(filepath=model_path, source=init_with)

if config.GPU_COUNT > 0:
    model.cuda()

# how to set epochs: di zeng !!!
# fang bian restart from ckpt !!!
# eg. epochs = {'heads': 5, 'all': 10}, all epoch is 10-5=5!!!

# Train the head branches
# 训练头部分支 冻结其他层
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train_model(dataset_train, dataset_val,
                  learning_rate=config.LEARNING_RATE,
                  epochs=5,
                  layers='heads')

# Fine tune all layers
# 精调所有层
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train_model(dataset_train, dataset_val,
                  learning_rate=config.LEARNING_RATE / 10,
                  epochs=10,
                  layers="all")

# Save weights
# 保存训练完成的权值和模型
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)


####################################
#      模型评估
####################################

print('\n####### 开始评估模型 ######\n')


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    EXP_DIR = exp_dir
    DETECTION_MIN_CONFIDENCE = 0.2


inference_config = InferenceConfig()
dataset_test = dataset_val
# Recreate the model in inference mode
model = HotRCNN(config=inference_config)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.get_ckpt_path()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path pointed to trained weights"
print("Loading weights from: ", model_path)
model.load_weights(filepath=model_path, source='ckpt')

if config.GPU_COUNT > 0:
    model.cuda()

# Test on a random image
image_id = np.random.choice(dataset_test.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask = dataset_test.load_image_gt(image_id, inference_config)

utils.log("original_image", image)
utils.log("image_meta", image_meta)
utils.log("gt_class_id", gt_class_id)
utils.log("gt_bbox", gt_bbox)
utils.log("gt_mask", gt_mask)

visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8))

results = model.detect(images=image[np.newaxis, :], metas=image_meta[np.newaxis, :])

r = results[0]  # 只有一张图片，一个检测结果
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            dataset_test.class_names, r['scores'], ax=utils.get_ax())

# 测试任意无GT-info图像
image = dataset_test.load_image(image_id)
results = model.detect([image])

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_test.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = dataset_test.load_image_gt(image_id, inference_config)
    # Run object detection
    results = model.detect(images=np.stack([image]), metas=np.stack([image_meta]))
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id, r["rois"], r["class_ids"], r["scores"])
    APs.append(AP)

mAP = np.mean(APs)
print("mAP: ", mAP)
