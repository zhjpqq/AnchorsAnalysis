# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = ' 15:09'

import sys
import os
import math
import random
import numpy as np
import scipy.misc
import skimage.color
import skimage.io
import cv2
import urllib.request
import shutil
import logging

import torch
from torch.utils import data
from PIL import Image

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from Configs.config import Config
from Utils import utils


class CocoDataset(data.Dataset):

    def __init__(self, config):
        """
        图片在本数据集中的索引 index，存放于 self._image_ids中。0,1,2,3，...。
        图片在COCO META中的索引 id, 存放于 self.image_info中。xxx,xxx,xxx,...。
        init load prepare iter/generator
        """
        self._image_ids = []
        self.image_info = []
        self.image_nums = None

        self.class_ids = []
        self.class_names = []
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.class_nums = None

        self.sources = []
        self.source_class_ids = {}
        self.class_from_source_map = {}

        self.external_to_class_id = {}
        self.external_to_image_id = {}

        self.config = config

    def __getitem__(self, image_index):
        """获取单张image数据，区别于generate获取一个batch的images数据"""

        image_id = self.image_ids[image_index]
        image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
            self.load_image_gt(image_id=image_id, config=self.config, augment=False)

        # 去空值
        # Skip images that have no instances. This can happen in cases
        # where we train on a subset of classes and the image doesn't
        # have any of the classes we care about.
        if not np.any(gt_class_ids > 0):
            return None

        # 去多余
        # If more instances than fits in the array, sub-sample from them.
        if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
            ids = np.random.choice(np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]

        # 去均值
        image = self.mold_image(image.astype(np.float32), self.config.MEAN_PIXEL)

        # Convert
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image_meta = torch.from_numpy(image_meta)
        gt_class_ids = torch.from_numpy(gt_class_ids)
        gt_boxes = torch.from_numpy(gt_boxes).float()
        gt_masks = torch.from_numpy(gt_masks.astype(int).transpose(2, 0, 1)).float()

        return image, image_meta, gt_class_ids, gt_boxes, gt_masks

    def __len__(self):
        return self.image_nums

    def __repr__(self):
        return 'Coco Dataset'

    @property
    def image_ids(self):
        return self._image_ids

    # 添加类信息
    def add_class(self, source, class_id, class_name):
        # 跳过已经添加的类
        for info in self.class_info:
            if info['source'] == source and info['id'] == class_id:
                return
        # 添加新出现的类
        self.class_info.append({'source': source,
                                'id': class_id,
                                'name': class_name})

    # 添加图片信息
    def add_image(self, image_source, image_id, image_path, **kwargs):
        image_info = {
            'id': image_id,
            'source': image_source,
            'path': image_path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    # 加载图片
    def load_image(self, image_id):
        image = skimage.io.imread(self.image_info[image_id]['path'])
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    # 加载掩膜
    def load_mask(self, image_id):
        """Load instance masks for the given image.
        一张图片上可能有多个物体，且分属不同的类，此处要使用源数据集中的类标id。
        image_id → (instance_mask，class_ID)×N

        不同数据集使用不同方法存储masks.
        本函数将其转换为统一格式:bitmap [height, width, instances]。

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            raise ValueError('muse be coco data!')

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        # 构造形为[height, weight, instance_count]的mask,且列出mask每个通道的class IDs
        for annotation in annotations:
            class_id = self.map_source_class_id("coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"], image_info["width"])
                # m: 2D numpy array [height, width] bool值
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                # 处理小物体标注
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                # 处理拥挤标注
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            mask = np.empty([0, 0, 0])
            class_ids = np.empty([0], dtype='int32')
            return mask, class_ids

    # 加载GT信息 GT_class_ids bbox mask
    def load_image_gt(self, image_id, config, augment=False):
        """image_id: 图片在数据集Meta信息中的真实ID"""
        image = self.load_image(image_id)
        mask, class_ids = self.load_mask(image_id)
        shape = image.shape
        image, window, scale, padding = self.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)
        mask = self.resize_mask(mask, scale, padding)

        # Random horizontal flips.
        if augment:
            if random.randint(0, 1):
                image = np.fliplr(image)
                mask = np.fliplr(mask)

        # Bounding boxes. Note that some boxes might be all zeros
        # if the corresponding mask got cropped out.
        # bbox: [num_instances, (y1, x1, y2, x2)]
        bbox = self.extract_bboxes(mask)

        # Active classes
        # Different datasets have different classes, so track the
        # classes supported in the dataset of this image.
        active_class_ids = np.zeros([self.class_nums], dtype=np.int32)
        source_class_ids = self.source_class_ids[self.image_info[image_id]["source"]]
        active_class_ids[source_class_ids] = 1

        # Resize masks to smaller size to reduce memory usage
        if config.USE_MINI_MASK:
            mask = self.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

        # Image meta data
        image_meta = self.compose_image_meta(image_id, shape, window, active_class_ids)

        return image, image_meta, class_ids, bbox, mask

    # 加载COCO数据，读取数据
    def load_coco(self, data_dir, subset, year, class_ids=None,
                   class_map=None, auto_download=None, **kwargs):

        if auto_download is True:
            self.auto_download(data_dir, subset, year)

        coco = COCO("{}/annotations/instances_{}{}.json".format(data_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(data_dir, subset, year)

        # Load all classes or a subset?
        # 添加需要的类id，所有类，还是其子集
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        # 添加需要的图片id，还是某些类构成的子集
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # 根据选择出的 class-ids 和 image-ids 添加更加详细的 class-info 和 image-info

        # Add classes 添加类信息 class_info
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images  添加图片信息 image_info
        for i in image_ids:
            self.add_image(
                image_source="coco",
                image_id=i,
                image_path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

        if 'return_coco' in kwargs.keys() and kwargs['return_coco']:
            return coco

    # 准备图片，以用于训练或测试，dict → list
    def prepare(self, class_map=None):
        """
        将来自不同数据集的图片进行整理
        :param class_map: 暂时不可用
        :return:
        """
        def clean_name(name):
            return ''.join(name.split(',')[:1])
        # 从信息字典中重构数据集
        self.class_nums = len(self.class_info)
        self.class_ids = np.arange(self.class_nums)
        self.class_names = [clean_name(c['name']) for c in self.class_info]
        self.sources = list(set([c['source'] for c in self.class_info]))
        self.image_nums = len(self.image_info)
        self._image_ids = np.arange(self.image_nums)

        # 将源数据集的类别ID与新数据集的类别ID进行关联，源数据集可能是多个不同数据集，
        # 如{"{VOC}.{1}":0,"{ImageNet}.{22}":1,"{COCO}.{9}":2,……}
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}

        # Map sources to class_ids they support
        # 将源数据集的源Name与新数据集的类别ID进行关联
        # 构建sources与class_ids之间的映射字典：source_class_ids{'source1':[],……}
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    # 映射 源数据集.类标 → 新数据集类标
    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    # 获取类id在源数据集中的id
    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source, '该类的source不匹配'
        return info['id']

    # The following two functions are from pycocotools with a few changes.
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        将多边形、压缩行程码RLE、行程码RLE统一转换为行程码RLE
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        将行程码格式的annotation转换为二值mask
        mask格式为：2D numpy数组 [height, wiodth] 0/1 bool值
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    # 图片网址链接
    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            raise ValueError('not coco dataset!')

    # 自动下载某类数据集
    def auto_download(self, dataDir, dataType, dataYear):
        """"""
        raise NotImplementedError('必须在子类中继承实现')

    # 数据生成器 generator
    def generate(self, config, batch_size=None, shuffle=None, augment=None):
        """
        参数自定义传入，或从config传入，train和val时的参数可能不相同
        :param config:
        :param batch_size:
        :param shuffle:
        :param augment:
        :return:
        """
        batch_size = config.BATCH_SIZE if batch_size is None else batch_size
        shuffle = config.TRAIN_VAL_SHUFFLE if shuffle is None else shuffle
        augment = config.TRAIN_VAL_AUGEMNT if augment is None else augment
        assert isinstance(batch_size, int) and batch_size >= 1, 'batch_size必须是正整数！'
        assert isinstance(shuffle, bool) and isinstance(augment, bool), 'shuffle和augment必须是布尔值！'

        b = 0  # batch item index
        image_index = -1
        image_ids = np.copy(self.image_ids)
        error_count = 0

        while True:
            try:
                # 逐张装填图片
                # 渐增image_index
                image_index = (image_index + 1) % len(image_ids)

                # 初始时随机排序
                if shuffle and image_index == 0:
                    np.random.shuffle(image_ids)

                # 获取GT box 和 mask
                image_id = image_ids[image_index]
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                    self.load_image_gt(image_id, config, augment=augment)

                # 跳过没有实例的图像。这种情况适用于：
                # 假设所有类的一个子集上训练，但某图片中的实例类都不在此子集中，则跳过此图片。
                if not np.any(gt_class_ids > 0):
                    continue

                # 初始化Batch数组
                if b == 0:
                    batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
                    batch_image_meta = np.zeros((batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                    batch_gt_class_ids = np.zeros((batch_size,) + config.MAX_GT_INSTANCES, dtype=np.int32)
                    batch_gt_boxes = np.zeros((batch_size,) + config.MAX_GT_INSTANCES, dtype=np.int32)
                    if not config.USE_MINI_MASK:
                        batch_gt_masks = np.zeros((batch_size, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1],
                                                   config.MAX_GT_INSTANCES))
                    else:
                        batch_gt_masks = np.zeros((batch_size, image.shape[0], image.shape[1], config.MAX_GT_INSTANCES))

                # 限制单张图片上的物体数量
                if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                    ids = np.random.choice(np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                    gt_class_ids = gt_class_ids[ids]
                    gt_boxes = gt_boxes[ids]
                    gt_masks = gt_masks[:, :, ids]

                # 将数据装入batch数组中 数量不足的部分全为0
                batch_images[b] = self.mold_image(image, config.MEAN_PIXEL)
                batch_image_meta[b] = image_meta
                batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
                batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
                batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks

                b += 1

                # 直到装满一个batchSize, 则yield返回一个迭代器
                # Batch full?  batchSize = GPU_COUNT*IMAGES_PER_GPU
                if b >= batch_size:
                    inputs = [batch_images, batch_image_meta,
                              batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                    outputs = []
                    yield inputs, outputs
                    # start a new batch
                    b = 0

            except GeneratorExit:
                raise Warning('数据生成器异常！')

            except KeyboardInterrupt:
                raise Warning('键盘终止！')

            except:
                # Log it and skip the image 其他异常
                logging.exception("Error processing image {}".format(self.image_info[image_id]))
                error_count += 1
                if error_count >= 1:
                    raise Warning('错误图片数量超过了1张!')

    ############################################################
    #  Data Formatting  数据格式化
    #
    #  通用静态方法，用于处理图像、盒子、掩膜、元信息，使得其符合格式要求.
    #
    #  image box mask meta
    ############################################################

    @staticmethod
    def compose_image_meta(image_id, image_shape, window, active_class_ids):
        """Takes attributes of an image and puts them in one 1D array. Use
        parse_image_meta() to parse the values back.
        将一张图片的信息放进一个1D数组中

        image_id: An int ID of the image. Useful for debugging.
        image_shape: [height, width, channels]
        window: (y1, x1, y2, x2) in pixels. The area of the image where the real
                image is (excluding the padding)    #图片上去除掉paading的部分，真正的图片部分
        active_class_ids: List of class_ids available in the dataset from which
            the image came. Useful if training on images from multiple datasets
            where not all classes are present in all datasets.
        """
        meta = np.array(
            [image_id] +  # size=1
            list(image_shape) +  # size=3
            list(window) +  # size=4 (y1, x1, y2, x2) in image cooredinates
            list(active_class_ids)  # size=num_classes
        )
        return meta

    @staticmethod
    # Two functions (for Numpy and TF) to parse image_meta tensors.
    def parse_image_meta(meta):
        """Parses an image info Numpy array to its components.
        See compose_image_meta() for more details.
        解析图片信息，返回元组
        """
        image_id = meta[:, 0]
        image_shape = meta[:, 1:4]
        window = meta[:, 4:8]  # (y1, x1, y2, x2) window of image in in pixels
        active_class_ids = meta[:, 8:]
        return image_id, image_shape, window, active_class_ids

    @staticmethod
    def parse_image_meta_graph(meta):
        """Parses a tensor that contains image attributes to its components.
        See compose_image_meta() for more details.
        解析图片信息，返回列表
        meta: [batch, meta length] where meta length depends on NUM_CLASSES
        """
        image_id = meta[:, 0]
        image_shape = meta[:, 1:4]
        window = meta[:, 4:8]
        active_class_ids = meta[:, 8:]
        return [image_id, image_shape, window, active_class_ids]

    @staticmethod
    def mold_image(images, rbg_mean):
        """Takes RGB images with 0-255 values and subtraces
        the mean pixel and converts it to float. Expects image
        colors in RGB order.
        在输入图片上去掉均值
        """
        return images.astype(np.float32) - rbg_mean

    @staticmethod
    def unmold_image(normalized_images, rbg_mean):
        """Takes a image normalized with mold() and returns the original."""
        return (normalized_images + rbg_mean).astype(np.uint8)

    @staticmethod
    def resize_image(image, min_dim=None, max_dim=None, padding=False):
        """
        Resizes an image keeping the aspect ratio.
        缩放图像，并保持形状比例

        min_dim: if provided, resizes the image such that it's smaller
            dimension == min_dim
        max_dim: if provided, ensures that the image longest side doesn't
            exceed this value.
        padding: If true, pads image with zeros so it's size is max_dim x max_dim

        Returns:
        image: the resized image
        window: (y1, x1, y2, x2). If max_dim is provided, padding might
            be inserted in the returned image. If so, this window is the
            coordinates of the image part of the full image (excluding
            the padding). The x2, y2 pixels are not included.
            填0之后的图像上的真正图像部分，保持原始图像部分的形状填零，原始图像小于缩放图像。
        scale: The scale factor used to resize the image
        padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
        """
        # Default window (y1, x1, y2, x2) and default scale == 1.
        h, w = image.shape[:2]
        window = (0, 0, h, w)
        scale = 1

        # 要么按最小边限定值缩放，要么按最大边限定值缩放
        # 当按最小边限制值缩放时，可能超过最大边限定值，则使用最大边限定值缩放
        # 边长最小不能小于多少，最大不能大于多少，不能同时满足时，按"大"约定走
        # Scale?
        if min_dim:
            # Scale up but not down
            scale = max(1, min_dim / min(h, w))
        # Does it exceed max dim?
        if max_dim:
            image_max = max(h, w)
            if round(image_max * scale) > max_dim:
                scale = max_dim / image_max
        # Resize image and mask
        if scale != 1:
            image = scipy.misc.imresize(
                image, (round(h * scale), round(w * scale)))
        # Need padding?
        if padding:
            # Get new height and width
            h, w = image.shape[:2]
            top_pad = (max_dim - h) // 2
            bottom_pad = max_dim - h - top_pad
            left_pad = (max_dim - w) // 2
            right_pad = max_dim - w - left_pad
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0)
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
        return image, window, scale, padding

    @staticmethod
    def resize_mask(mask, scale, padding):
        """Resizes a mask using the given scale and padding.
        Typically, you get the scale and padding from resize_image() to
        ensure both, the image and the mask, are resized consistently.

        从resize_image()中获取scale和padding参数，确保image和mask的一致性

        scale: mask scaling factor
        padding: Padding to add to the mask in the form
                [(top, bottom), (left, right), (0, 0)]
        """
        h, w = mask.shape[:2]
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)  # 最近邻插值
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
        return mask

    @staticmethod
    def minimize_mask(bbox, mask, mini_shape):
        """Resize masks to a smaller version to cut memory load.
        Mini-masks can then resized back to image scale using expand_masks()

        将mask缩放到小尺寸以节约内存，可以 expand_masks()返回去

        在掩膜上按bbox大小裁切 → 再将裁切块缩放到mini_shape

        See inspect_data.ipynb notebook for more details.
        """
        mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            y1, x1, y2, x2 = bbox[i][:4]
            m = m[y1:y2, x1:x2]
            if m.size == 0:
                raise Exception("Invalid bounding box with area of zero")
            m = scipy.misc.imresize(m.astype(float), mini_shape, interp='bilinear')
            mini_mask[:, :, i] = np.where(m >= 128, 1, 0)
        return mini_mask

    @staticmethod
    def expand_mask(bbox, mini_mask, image_shape):
        """Resizes mini masks back to image size. Reverses the change
        of minimize_mask().

        See inspect_data.ipynb notebook for more details.
        """
        mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
        for i in range(mask.shape[-1]):
            m = mini_mask[:, :, i]
            y1, x1, y2, x2 = bbox[i][:4]
            h = y2 - y1
            w = x2 - x1
            m = scipy.misc.imresize(m.astype(float), (h, w), interp='bilinear')
            mask[y1:y2, x1:x2, i] = np.where(m >= 128, 1, 0)
        return mask

    # TODO: Build and use this function to reduce code duplication
    @staticmethod
    def mold_mask(mask, config):
        pass

    @staticmethod
    def unmold_mask(mask, bbox, image_shape):
        """Converts a mask generated by the neural network into a format similar
        to it's original shape.
        mask: [height, width] of type float. A small, typically 28x28 mask.
        bbox: [y1, x1, y2, x2]. The box to fit the mask in.

        Returns a binary mask with the same size as the original image.
        """
        threshold = 0.5
        y1, x1, y2, x2 = bbox
        mask = scipy.misc.imresize(
            mask, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0
        mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

        # Put the mask in the right location.
        full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask
        return full_mask

    @staticmethod
    def extract_bboxes(mask):
        """Compute bounding boxes from masks.
        从masks中计算boxes，面积转为角点坐标

        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """
        boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = np.array([y1, x1, y2, x2])
        return boxes.astype(np.int32)

    def build_coco_results(self, image_ids, class_ids, scores, rois, masks):
        """Arrange resutls to match COCO specs in http://cocodataset.org/#format
            按COCO数据集的官方格式，重新整理检测结果 -> rois
        """
        # If no results, return an empty list
        if rois is None:
            return []

        results = []
        for image_id in image_ids:
            # Loop through detections
            for i in range(rois.shape[0]):
                class_id = class_ids[i]
                score = scores[i]
                bbox = np.around(rois[i], 1)  # 检测出的roi的坐标可能是小数,精确到1位小数点
                mask = masks[:, :, i]

                result = {
                    "image_id": image_id,
                    "category_id": self.get_source_class_id(class_id, "coco"),
                    "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                    "score": score,
                    "segmentation": maskUtils.encode(np.asfortranarray(mask))
                }
                results.append(result)
        return results

    # 增加数据
    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i