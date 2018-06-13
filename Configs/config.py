# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = ' 10:05'

import math
import numpy as np
import os
import sys


class Config(object):
    # 数据集名称，coco/voc...
    NAME = None

    # 实验路径
    # ROOT_DIR/EXP_DIR/LOG_DIR/xxx.ckpt
    # 比如： HRCNN/ Experiments/coco_exp/ coco_exp_1/xxx.ckpt
    ROOT_DIR = None
    EXP_DIR = None
    LOG_DIR = None

    # ######################################################
    #    GPU 参数
    # ######################################################

    GPU_COUNT = 1

    IMAGES_PER_GPU = 2

    # ######################################################
    #    Backbone 参数
    # ######################################################
    BACKBONE_DIR = None

    BACKBONE_NAME = None

    BACKBONE_ARCH = None  # 'resnet50', 'resnet101', 'resnext101', 'vgg16'

    BACKBONE_INIT = False  # True will auto download & load weights when create backone

    # Backone Stages :
    # [image/1024 -->> conv1/bn1/relu->512, maxpool->256, layer1->256, layer2->128, layer3->64, layer4->32, avgpool->26]
    # [1024, 512, 256, 128, 64, 32, 26] -->> [1, 2, 4, 8, 16, 32, 39.38]

    # [stage1/512: conv1/bn1/relu; stage2/256:maxpool,layer1; stage3/128/layer2; stage4/64/layer3; stage5/32/layer4;
    # stage6/39.38/avgpool]  when image shape is 1024

    # [stage1/2: conv1/bn1/relu; stage2/4:maxpool,layer1; stage3/8/layer2; stage4/16/layer3; stage5/32/layer4;
    # stage6/39.38/avgpool] fixed strides of resnet

    # in fpn, stage 6 is downsample from stage5 by 1/2!
    BACKBONE_INCLUDE = None  # 指定可作为特征提取器使用的layer

    BACKBONE_STRIDES = [2, 4, 8, 16, 32, 64]  # 输出特征图相对于原图的尺寸比例

    BACKBONE_CHANNELS = [32, 64, 128, 256, 512, 1024, 2048]  # 输出特征图的通道数

    BACKBONE_SHAPES = None  # = IMAGE_SHAPE/BACKBONE_STRIDES

    # ######################################################
    #   Feature Fusion 参数，控制特征融合
    # ######################################################
    # fpn: channel 改变
    # lsc: channel shapes 改变

    FUSION_METHOD = ['fpn', 'lsc', 'ssd', 'simple', 'none'][4]

    FUSION_LEVELS = [1, 5][0]  # 融合之后的特征级数

    FUSION_CHANNELS_IN = BACKBONE_CHANNELS  # 融合之前的特征通道数

    FUSION_CHANNELS_OUT = 256  # 融合之后的特征通道数

    FUSION_STRIDES = np.array(BACKBONE_STRIDES)[4]  # 融合之后原图与特征图的尺寸比    # same to Backbone strides

    FUSION_SHAPES = None

    # FPN 融合方案

    # LSC 融合方案
    LSC_KERNEL_SIZE = 16

    # ######################################################
    #     DataSet参数， 控制数据加载/预处理
    # ######################################################
    CLASSES_NUMS = 1 + 0

    # resnet backbone for imagenet is (224, 224)
    # todo??? shape=[MIN, MAX] or [MAX, MIN]
    IMAGE_MIN_DIM = 800

    IMAGE_MAX_DIM = 1024

    IMAGE_SHAPE = np.array([IMAGE_MIN_DIM, IMAGE_MAX_DIM, 3])

    IMAGE_PADDING = True

    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    TRAIN_AUGMENT = True

    VAL_AUGMENT = False

    TRAIN_VAL_AUGEMNT = True

    TRAIN_SHUFFLE = True

    VAL_SHUFFLE = True

    TRAIN_VAL_SHUFFLE = True

    # ######################################################
    #      BP参数， 控制反向传播
    # ######################################################

    TRAIN_STEPS_PER_EPOCH = 1000

    VAL_STEPS_PER_EPOCH = 50

    LEARNING_RATE = 0.001

    LEARNING_MOMENTUM = 0.9

    WEIGHT_DECAY = 0.0001

    # ######################################################
    #      GT-Class-ID    GT-BBox  GT-MASK
    # ######################################################

    MAX_GT_INSTANCES = 100  # 每张图片上允许的最多GT实例

    USE_MINI_MASK = False

    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # ######################################################
    #           Anchors
    # ######################################################

    # 每张图片上的Anchors参数 由(level, Loc, Scale, Ratio)决定

    ANCHORS_PER_IMAGE = 1000

    ANCHOR_SCALES = (32, 64, 128, 256, 512)

    ANCHOR_ASPECTS = [1 / 2, 1, 2]

    ANCHOR_STRIDE = 1  # 锚点生成间隔，每隔几个点生成一个锚点，控制锚点的稀疏/密集程度

    ANCHOR_LEVELS = FUSION_LEVELS  # 锚点位于几个level的金字塔上，多个level的时候与Scales相等

    ANCHOR_ZERO_AREA = 1 * 1

    ANCHOR_GTBOX_IOU = (0.7, 0.5)  # (positive, negetive), deprecated, replaced by ROIS_GTBOX_IOU

    # ######################################################
    #           Proposals
    # ######################################################
    PROPOSALS_PER_IMAGE = 1000

    # ######################################################
    #           ROIs & ROI-Target & ROI-Transform
    # ######################################################
    TRAIN_ROIS_PER_IMAGE = 500  # 所有用于训练的RoIs

    ROIS_POSITIVE_RATIO = 0.33  # 其中正样本占的比例

    ROIS_GTBOX_IOU = (0.88, 0.60)  # ROI与GTbox的交叠阈值，判断±ROIs  (positive MAX threshold, negetive MIN threshold)

    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    BBOX_POOL_SIZE = (7, 7)

    MASK_POOL_SIZE = (14, 14)

    MASK_SHAPE = [28, 28]

    # ######################################################
    #          Class - BBOX - Mask
    # ######################################################

    CLASS_BBOX_METHOD = ['mask-rcnn', 'light-head'][0]

    MASK_HEAD_METHOD = ['mask-rcnn', 'light-head'][0]

    # ######################################################
    #           Detections 参数
    # ######################################################
    # 每张图片允许的最大检出量
    DETECTION_MAX_INSTANCES = 100

    # 接受一个ROIs的最小置信概率，低于此值将被略过
    # 用于过滤proposals → refine_detections()
    DETECTION_MIN_CONFIDENCE = 0.7

    # 对同一类的所有检出进行非极大值抑制
    # 用于过滤proposals → refine_detections()
    DETECTION_NMS_THRESHOLD = 0.3

    # ######################################################
    #           Evaluation 参数
    # ######################################################
    HRCNN_MODEL_PATH = None  # 已经训练完成的HRCNN模型的路径

    def __init__(self):

        # train batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # backbone 各个stage的image大小
        # 1024/[4, 8, 16, 32, 64] = [256, 128, 64, 32, 16]
        if type(self.BACKBONE_STRIDES) != list:
            self.BACKBONE_STRIDES = [self.BACKBONE_STRIDES]
        self.BACKBONE_SHAPES = np.ceil(np.array([[self.IMAGE_SHAPE[0] / stride, self.IMAGE_SHAPE[1] / stride]
                                                 for stride in self.BACKBONE_STRIDES])).astype(np.int)

        # fusion map shapes
        if type(self.FUSION_STRIDES) != list:
            self.FUSION_STRIDES = [self.FUSION_STRIDES]
        self.FUSION_SHAPES = np.ceil(np.array([[self.IMAGE_SHAPE[0] / stride, self.IMAGE_SHAPE[1] / stride]
                                               for stride in self.FUSION_STRIDES])).astype(np.int)

        if self.BACKBONE_ARCH.startwith('resnet'):
            self.BACKBONE_INCLUDE = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'][0:-5]

        self.check()

    def check(self, verbose=0):
        """
        FUSION 承前启后，需要对前后约束关系进行检查, Backbone & Fusion & Anchors
        """
        # 自约束
        assert len(self.BACKBONE_STRIDES) == len(self.BACKBONE_CHANNELS), 'assert 2'
        assert len(self.FUSION_STRIDES) == len(self.FUSION_SHAPES), 'assert 3'

        # 关联约束
        if self.FUSION_LEVELS == 1:
            # fusion & backbone 约束
            assert self.FUSION_LEVELS == len(self.BACKBONE_CHANNELS), 'assert'
            assert self.FUSION_LEVELS == len(self.BACKBONE_STRIDES), 'assert'
            # fusion & 自身约束
            assert self.FUSION_LEVELS == len(self.FUSION_STRIDES), 'assert 2'
            assert self.FUSION_METHOD in ['simple', 'none'], 'assert 6'
            # fusion & ANCHORS 约束
            assert self.FUSION_LEVELS == self.ANCHOR_LEVELS, 'assert 1'

        elif self.FUSION_LEVELS > 1:
            # fusion & backbone 约束
            assert self.FUSION_LEVELS == len(self.BACKBONE_CHANNELS), 'assert'
            assert self.FUSION_LEVELS == len(self.BACKBONE_STRIDES), 'assert'
            # fusion & 自身约束
            assert self.FUSION_LEVELS == len(self.FUSION_STRIDES), 'assert 2'
            assert self.FUSION_METHOD in ['fpn', 'ssd'], 'assert 6'
            # fusion & ANCHORS 约束
            assert self.FUSION_LEVELS == self.ANCHOR_LEVELS, 'assert 1'
            assert self.FUSION_LEVELS == len(self.ANCHOR_SCALES), 'assert 5'

        return True

    def display(self):
        pass


class CocoConfig(Config):
    NAME = 'coco'
    IMAGES_PER_GPU = 2
    GPU_COUNT = 2
    CLASSES_NUMS = 1 + 80


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    CLASSES_NUMS = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024  # 224
    IMAGE_MAX_DIM = 1024  # 224

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    TRAIN_STEPS_PER_EPOCH = 1

    # use small validation steps since the epoch is small
    VAL_STEPS_PER_EPOCH = 1
