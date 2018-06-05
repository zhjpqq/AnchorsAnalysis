# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = ' 10:05'

import math
import numpy as np
import os
import sys


class Config(object):

    # 当前数据集/实验的名称
    NAME = 'coco'

    # dang qian shiyan de lujin
    EXP_DIR = ''

    # GPU 数量
    GPU_COUNT = 1

    IMAGES_PER_GPU = 2

    # 骨干网络参数
    BACKBONE_ARCH = ['resnet50', 'resnet101', 'resnext101', 'vgg16'][0]

    BACKBONE_DIR = None

    BACKBONE_PATH = None    # 'point to Backbones file path'

    BACKBONE_PreTRAINED = False    # True will auto download & load

    # Backone Stages :
    # [image/1024 -->> conv1/bn1/relu->512, maxpool->256, layer1->256, layer2->128, layer3->64, layer4->32, avgpool->26]
    # [1024, 512, 256, 128, 64, 32, 26] -->> [1, 2, 4, 8, 16, 32, 39.38]

    # [stage1/512: conv1/bn1/relu; stage2/256:maxpool,layer1; stage3/128/layer2; stage4/64/layer3; stage5/32/layer4;
    # stage6/39.38/avgpool]  when image shape is 1024

    # [stage1/2: conv1/bn1/relu; stage2/4:maxpool,layer1; stage3/8/layer2; stage4/16/layer3; stage5/32/layer4;
    # stage6/39.38/avgpool] fixed strides of resnet

    # in fpn, stage 6 is downsample from stage5 by 1/2!
    BACKBONE_INCLUDE = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'][0:-5]

    BACKBONE_STRIDES = [2, 4, 8, 16, 32, 64]

    # 特征融合参数
    FEATURE_FUSION_METHOD = ['fpn', 'lsc', 'ssd', 'simple', 'none'][4]

    FEATURE_FUSION_LEVELS = [1, 5][0]   # 融合之后的特征层级数

    FEATURE_FUSION_CHANNELS = 256      # 融合之后的特征通道数

    # same to Backbone strides
    FEATURE_FUSION_STRIDES = 1/np.array([2, 4, 8, 16, 32, 64])[4]    # 融合之后原图与特征图的尺寸比

    # FPN

    # LSC
    LSC_IN_CHANNELS = 2048
    LSC_OUT_CHANNELS = FEATURE_FUSION_CHANNELS
    LSC_KERNEL_SIZE = 16

    # # # 图片输入参数 # # #
    CLASSES_NUMS = 1 + 0

    # resnet backbone for imagenet is (224, 224)
    IMAGE_MIN_DIM = 800

    IMAGE_MAX_DIM = 1024

    IMAGE_PADDING = True

    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # # # 图片扩增参数 # # #

    TRAIN_AUGMENT = True

    VAL_AUGMENT = False

    TRAIN_VAL_AUGEMNT = True

    TRAIN_SHUFFLE = True

    VAL_SHUFFLE = True

    TRAIN_VAL_SHUFFLE = True

    # # # 训练/验证参数 # # #
    TRAIN_STEPS_PER_EPOCH = 1000

    VAL_STEPS_PER_EPOCH = 50

    LEARNING_RATE = 0.001

    LEARNING_MOMENTUM = 0.9

    WEIGHT_DECAY = 0.0001

    # # # Anchors-Proposals-Rois 检测参数 # # #

    # 关于不同阶段的数量
    # Anchors → Proposals → ROIS → Train_ROIs → { +ROIs, -ROIs }
    # 保持正负样本比例不变

    # 每张图片上的 锚点框/建议区/兴趣区/GT区 的最大限制
    # ANCHORS_PER_IMAGE = Anchors x len(scales) x len(ratios) - zero_area_boxes
    ANCHORS_PER_IMAGE = 1000        # 基于点热度

    PROPOSALS_PER_IMAGE = 1000      # 基于领域热度

    TRAIN_ROIS_PER_IMAGE = 200      # 所有用于训练的RoIs

    ROIS_POSITIVE_RATIO = 0.33      # 其中正样本占的比例

    MAX_GT_INSTANCES = 100      # 只用于datagenerator中，指明gt的数量

    # MAX_GT_INSTANCES < TRAIN_ROIS_PER_IMAGE ?? 100 < 200*0.33=66

    # Bbox精调标准差
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # 每张图片上的Anchors参数
    # (Loc, Scale, Ratio)
    # ANCHOR_COUNTS = 500  # 就是 ANCHORS_PER_IMAGE

    # [(32, 64, 128, 256, 512), (32, 64, 128, 256), (32, 64, 128), (32, 64), (32,)]
    ANCHOR_SCALES = (32, 64, 128, 256, 512)

    ANCHOR_ASPECTS = [1/2, 1, 2]

    ANCHOR_LEVEL_NUMS = [1, 5][0]        # 锚点位于几个level的金字塔上，多个level的时候与Scales相等

    ANCHOR_HEAT_METHOD = ['accumulate', 'separable', 'window'][0]  # 热度值统计方法

    ANCHOR_ZERO_AREA = 1*1

    ANCHOR_GTBOX_IOU = (0.88, 0.88)   # (positive, negetive)

    # class-bbox-mask 头
    CLASS_BBOX_METHOD = ['mask-rcnn', 'light-head'][0]

    MASK_HEAD_METHOD = ['mask-rcnn', 'light-head'][0]

    BBOX_POOL_SIZE = (7, 7)

    MASK_POOL_SIZE = (14, 14)

    MASK_SHAPE = [28, 28]

    # 是否使用mini mask
    USE_MINI_MASK = False

    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # 每张图片的备选检出量
    # 再经由下面的CONFINDENCE过滤之后，即得到每张图片上的最终检出结果
    DETECTION_MAX_INSTANCES = 100

    # 接受一个ROIs的最小置信概率，低于此值将被略过
    # 用于过滤proposals → refine_detections()
    DETECTION_MIN_CONFIDENCE = 0.7

    # 对detections进行非极大值抑制
    # 用于过滤proposals → refine_detections()
    # 对同一类的所有检出进行非极大值抑制
    DETECTION_NMS_THRESHOLD = 0.3

    # 已经训练完成的HRCNN模型的路径
    HRCNN_MODEL_PATH = None

    def __init__(self):

        # train batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU*self.GPU_COUNT

        # input image shapes
        self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # backbone 各个stage的image大小
        # 1024/[4, 8, 16, 32, 64] = [256, 128, 64, 32, 16]
        self.BACKBONE_SHAPES = np.array([[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
                                          int(math.ceil(self.IMAGE_SHAPE[1] / stride))]
                                         for stride in self.BACKBONE_STRIDES])

        # fusion map shapes
        if not isinstance(self.FEATURE_FUSION_STRIDES, list):
            self.FEATURE_FUSION_STRIDES = [self.FEATURE_FUSION_STRIDES]
        self.FUSION_SHAPES = np.array([[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
                                        int(math.ceil(self.IMAGE_SHAPE[1] / stride))]
                                       for stride in self.FEATURE_FUSION_STRIDES])

        self.check()

    def check(self):
        """"""
        assert self.FEATURE_FUSION_LEVELS == self.ANCHOR_LEVEL_NUMS, 'assert 1'
        assert len(self.FEATURE_FUSION_STRIDES) == self.ANCHOR_LEVEL_NUMS, 'assert 2'

        # 融合后特征级数levels>1时，必须与anchors数量一致
        if self.FEATURE_FUSION_LEVELS > 1:
            assert self.FEATURE_FUSION_LEVELS == len(self.ANCHOR_SCALES), 'assert 5'
            assert self.FEATURE_FUSION_METHOD in ['fpn', 'ssd'], 'assert 6'
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
    IMAGE_MIN_DIM = 1024    #224
    IMAGE_MAX_DIM = 1024    #224

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    TRAIN_STEPS_PER_EPOCH = 1

    # use small validation steps since the epoch is small
    VAL_STEPS_PER_EPOCH = 1