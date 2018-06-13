#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/4/3 11:27'
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

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# from distutils.version import LooseVersion
# assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
# assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

from Configs.config import CocoConfig
from DataSets.coco_dataset import CocoDataset
from Models.hot_rcnn import HotRCNN
from Utils import utils


# 关于目录设定规则

# 路径顺序：Project_Root_Dir / Exp_Dir / log_dir / checkpoint.h5 + events.os
# 工程目录 root_dir / 实验目录 exp_dir / 日志目录 log_dir / ckpt.h5 + events.os
# log_dir下有2种文件：checkpoint.h5（模型） ，events.os（日志）
# resent.h5 hrcnn.h5 属于已经完全训练好的模型，直接放在 实验目录exp_dir 下面

curr_dir = os.getcwd()
root_dir = os.path.dirname(curr_dir)
exp_dir = os.path.join(root_dir, 'Experiments', 'coco_exp')
log_dir = os.path.join(exp_dir, '**由函数self.set_log_dir()动态生成！**')

# pretrained models directory
# 训练完成的 resnet，hrcnn模型路径
# 训练过程中的 checkpoint.h5 模型保存在log_dir中: root_dir/exp_dir/log_dir/ckpt.h5
backbone_dir = os.path.join(root_dir, 'Backbones')
backbone_name = 'resnet50-19c8e357.pth'
backbone_path = os.path.join(backbone_dir, backbone_name)
hrcnn_model_path = os.path.join(exp_dir, 'hrcnn_mode_xxx.h5')
hrcnn_ckpt_path = os.path.join(exp_dir, log_dir, "hot_rcnn_ckpt_xxx.h5")


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    封装了COCO官方的评估函数: COCOeval()
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    # 取出该图片在COCO中的原始ID
    # 内外两层id：第1个id是二次封装后的id，第2个"id"是在源数据集中的id
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = CocoDataset.load_image(image_id)

        # Run detection, 运行检测, 只送入了一张图片[image], 所以检测结果也是一个result=[result][0]
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # 一张图片上的检测结果包含 [rois, class_ids, scores, masks]
        image_results = CocoDataset.build_results(coco_image_ids[i:i + 1],
                                                  r["class_ids"], r["scores"], r["rois"], r["masks"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    # 将检测结果转换为COCO指定格式
    coco_results = coco.loadRes(results)

    # Evaluate
    # 对检测结果，运行COCO评估
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='训练和评估coco，HotRCNN')
    parser.add_argument('command',
                        metavar='<command>',
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar='path/to/coco',
                        help='Directory of the MS-COCO dataset',
                        default='')
    parser.add_argument('--year', required=False,
                        default=2014,
                        metavar='<year>',
                        help='coco data year')
    parser.add_argument('--model', required=False,
                        # default='hot_rcnn_coco.h5',
                        metavar="/path/to/weights.h5",
                        help="预训练模型的位置，用于存放resnet或hrcnn模型")
    parser.add_argument('--exp', required=False,
                        default=exp_dir,
                        metavar="/path/to/exp/",
                        help='日志文件夹(存放events和ckpt)的上级目录，/path/to/exp/logs/(events.os+ckpt.h5)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Experiments: ", args.exp)
    print("Auto Download: ", args.download)

    # Config 参数配置
    config = CocoConfig()
    config.BACKBONE_DIR = backbone_dir
    config.BACKBONE_PATH = backbone_path
    config.HRCNN_MODEL_PATH = hrcnn_model_path
    if args.command == 'evaluate':
        config.GPU_COUNT = 1
        config.IMAGES_PER_GPU = 1
        config.DETECTION_MIN_CONFIDENCE = 0
    config.display()

    # Model 模型创建
    model = HotRCNN(config=config)

    # Weights 权值加载
    # 1. 未训练加载：从imagenet加载resnet模型的权值
    # 2. 已训完加载：加载一个已经训练完成的完整hrcnn模型
    # 3. 半训练加载：从最近的训练节点加载一个半成品hrcnn模型
    init_with = args.model.lower()
    if init_with == 'backbone':
        model_path = model.get_backone_path()
    elif init_with == 'hrcnn':
        model_path = model.get_weights_path('hrcnn', path='')
    elif init_with == 'ckpt':
        model_path = model.get_ckpt_path()[1]
    else:
        model_path = None
    assert model_path is not None, 'No Model File is Found!'

    print('开始加载预训练权值 ……', model_path)
    model.load_weights(filepath=model_path, source=init_with)

    if config.GPU_COUNT > 0:
        model.cuda()

    # Train or Evaluate 训练或测试
    if args.command == 'train':

        # 构造coco数据集
        dataset_train = CocoDataset(config=config)
        dataset_train.load_coco(data_dir=args.dataset,
                                subset='train',
                                year=args.year,
                                return_coco=True,
                                auto_download=args.download)
        dataset_train.load_coco(data_dir=args.dataset,
                                subset='valminusminival',
                                year=args.year,
                                return_coco=True,
                                auto_download=args.download)
        dataset_train.prepare()

        dataset_val = CocoDataset(config=config)
        dataset_val.load_coco(data_dir=args.dataset,
                              subset='minival',
                              year=args.year,
                              return_coco=True,
                              auto_download=args.download)
        dataset_val.prepare()

        # 训练开始

        # stage1 训练头部
        print("Training network heads")
        model.train_model(dataset_train, dataset_val,
                          learning_rate=config.LEARNING_RATE,
                          epochs=40,
                          layers='heads')

        # stage2 训练4+层
        print("Training network 4+ layers")
        model.train_model(dataset_train, dataset_val,
                          learning_rate=config.LEARNING_RATE,
                          epochs=120,
                          layers='4+')

        # stage3 训练全部层
        print("Training network all layers")
        model.train_model(dataset_train, dataset_val,
                          learning_rate=config.LEARNING_RATE / 10,
                          epochs=160,
                          layers='all')

    elif args.command == 'evaluate':
        # 构造coco数据集
        dataset_val = CocoDataset(config=config)
        coco = dataset_val.load_coco(data_dir=args.dataset,
                                     subset='minival',
                                     year=args.year,
                                     return_coco=True,
                                     auto_download=args.download)
        dataset_val.prepare()
        # 开始运行评估
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))

    else:
        print("'{}' is not recognized. Use 'train' or 'evaluate'".format(args.command))

