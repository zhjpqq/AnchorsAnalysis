# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = ' 10:05'

import math
import numpy as np
import os
import sys


class Config(object):
    """
    when cut between levels=1 & levels=5,
    you should change bellow parameters.

    BACKBONE_STRIDES, BACKBONE_CHANNELS, BACKBONE_STAGES
    FUSION_METHOD, FUSION_LEVELS, FUSION_CHANNELS_OUT
    """

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
    BACKBONE_DIR = '/HRCNN/Backbones'

    BACKBONE_NAME = 'resnet50-19c8e357.pth'

    BACKBONE_ARCH = 'resnet50'  # 'resnet50', 'resnet101', 'resnext101', 'vgg16'

    BACKBONE_INIT = False  # True will auto download & load weights when create backone

    # Backone Stages :
    # [image/1024 -->> conv1/bn1/relu->512, maxpool->256, layer1->256, layer2->128, layer3->64, layer4->32, avgpool->26]
    # [1024, 512, 256, 128, 64, 32, 26] -->> [1, 2, 4, 8, 16, 32, 39.38]

    # [stage1/512: conv1/bn1/relu; stage2/256:maxpool,layer1; stage3/128/layer2; stage4/64/layer3; stage5/32/layer4;
    # stage6/39.38/avgpool]  when image shape is 1024

    # [stage1/2: conv1/bn1/relu; stage2/4:maxpool,layer1; stage3/8/layer2; stage4/16/layer3; stage5/32/layer4;
    # stage6/39.38/avgpool] fixed strides of resnet

    # bellow 3 parameters should tong yi !!!
    # this is the settings for resnet50 / resnet101 .
    BACKBONE_SHAPES = None  # = IMAGE_SHAPE/BACKBONE_STRIDES

    # level=5 [1:] & level=1 [3] for yixia 3 ge..
    option = 3
    BACKBONE_STAGES = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6'][option]     # feature maps of stage

    BACKBONE_STRIDES = [2, 4, 8, 16, 32, 64][option]  # 输出特征图相对于原图的尺寸比例, 64: P6 not C6

    BACKBONE_CHANNELS = [64, 256, 512, 1024, 2048, 2048][option]       # 输出特征图的通道数

    # ######################################################
    #   Feature Fusion 参数，控制特征融合
    # ######################################################
    # fpn: channel 改变
    # lsc: channel shapes 改变

    # level=5 [fpn] & level=1 [lsc, none]
    FUSION_METHOD = ['fpn', 'lsc', 'ssd', 'simple', 'none'][-1]

    FUSION_LEVELS = [1, 5][0]  # 融合之后的特征级数

    FUSION_CHANNELS_IN = BACKBONE_CHANNELS  # 融合之前的特征通道数

    # level=5 [512 or customVal] & level=1 [== BACKBONE_CHANNELS]
    FUSION_CHANNELS_OUT = BACKBONE_CHANNELS  # 融合之后的特征通道数

    FUSION_STRIDES = BACKBONE_STRIDES  # 融合之后原图与特征图的尺寸比    # same to Backbone strides

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
    IMAGE_MIN_DIM = 256

    IMAGE_MAX_DIM = 256

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
    # scale is respect to original image shape

    # levels=1: 1000, levels>1: (16*16)*256 > 1000
    ANCHORS_PER_IMAGE = 16*16

    ANCHOR_SCALES = [512, 256, 128, 64, 32]

    ANCHOR_ASPECTS = [1 / 2, 1, 2]

    ANCHOR_STRIDE = 1  # 锚点生成间隔，每隔几个点生成一个锚点，控制锚点的稀疏/密集程度

    ANCHOR_HEAT_METHOD = ['accumulate', 'separable', 'window'][0]

    ANCHOR_LEVELS = FUSION_LEVELS  # 锚点位于几个level的金字塔上，多个level的时候与Scales相等

    ANCHOR_ZERO_AREA = 1 * 1

    ANCHOR_GTBOX_IOU = (0.7, 0.5)  # (positive, negetive), deprecated, replaced by ROIS_GTBOX_IOU

    ANCHOR_METHOD = ['general', 'hotpoint'][1]

    # ######################################################
    #           Proposals
    # ######################################################
    PROPOSALS_METHOD = ['random', 'hotproposal'][0]

    PROPOSALS_PER_IMAGE = 256

    # ######################################################
    #           ROIs & ROI-Target & ROI-Transform
    # ######################################################
    TRAIN_ROIS_PER_IMAGE = 200  # 所有用于训练的RoIs

    ROIS_POSITIVE_RATIO = 0.33  # 其中正样本占的比例

    ROIS_GTBOX_IOU = (0.88, 0.77)  # ROI与GTbox的交叠阈值，判断±ROIs  (positive MAX threshold, negetive MIN threshold)

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

        if type(self.BACKBONE_STAGES) != list:
            self.BACKBONE_STAGES = [self.BACKBONE_STAGES]

        if type(self.BACKBONE_CHANNELS) != list:
            self.BACKBONE_CHANNELS = [self.BACKBONE_CHANNELS]

        if type(self.BACKBONE_STRIDES) != list:
            self.BACKBONE_STRIDES = [self.BACKBONE_STRIDES]

        if type(self.FUSION_STRIDES) != list:
            self.FUSION_STRIDES = [self.FUSION_STRIDES]

        if type(self.FUSION_CHANNELS_IN) != list:
            self.FUSION_CHANNELS_IN = [self.FUSION_CHANNELS_IN]

        # input image shape
        self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MAX_DIM, 3])

        # backbone 各个stage的fmap大小
        # 1024/[4, 8, 16, 32, 64] = [256, 128, 64, 32, 16]
        self.BACKBONE_SHAPES = np.ceil(np.array([[self.IMAGE_SHAPE[0] / stride, self.IMAGE_SHAPE[1] / stride]
                                                 for stride in self.BACKBONE_STRIDES])).astype(np.int)

        # fusion 各个stage的fmap大小
        self.FUSION_SHAPES = np.ceil(np.array([[self.IMAGE_SHAPE[0] / stride, self.IMAGE_SHAPE[1] / stride]
                                               for stride in self.FUSION_STRIDES])).astype(np.int)


        self.check()

    def check(self, verbose=0):
        """
        FUSION 承前启后，需要对前后约束关系进行检查, Backbone & Fusion & Anchors
        """
        # 自约束
        assert len(self.BACKBONE_STAGES) == len(self.BACKBONE_CHANNELS), 'assert 2'
        assert len(self.BACKBONE_STAGES) == len(self.BACKBONE_STRIDES), 'assert 2'
        assert len(self.FUSION_STRIDES) == len(self.FUSION_SHAPES), 'assert 3'

        assert self.ANCHORS_PER_IMAGE >= self.PROPOSALS_PER_IMAGE
        assert self.PROPOSALS_PER_IMAGE >= self.TRAIN_ROIS_PER_IMAGE
        assert self.TRAIN_ROIS_PER_IMAGE > self.DETECTION_MAX_INSTANCES

        # 关联约束
        if self.FUSION_LEVELS == 1:
            # fusion & backbone 约束
            assert self.FUSION_LEVELS == len(self.BACKBONE_STAGES), 'assert'
            # fusion & 自身约束
            assert self.FUSION_LEVELS == len(self.FUSION_STRIDES), 'assert 2'
            assert self.FUSION_METHOD in ['simple', 'none'], 'assert 6'
            # fusion & ANCHORS 约束
            assert self.FUSION_LEVELS == self.ANCHOR_LEVELS, 'assert 1'

        elif self.FUSION_LEVELS > 1:
            # fusion & backbone 约束
            assert self.FUSION_LEVELS == len(self.BACKBONE_STAGES), 'assert'
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
    IMAGES_PER_GPU = 1
    GPU_COUNT = 2
    CLASSES_NUMS = 1 + 90


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
    TRAIN_ROIS_PER_IMAGE = 300

    # Use a small epoch since the data is simple
    TRAIN_STEPS_PER_EPOCH = 1

    # use small validation steps since the epoch is small
    VAL_STEPS_PER_EPOCH = 1
