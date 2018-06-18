#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/4/7 21:22'
__author__ = 'ooo'

import sys
import os
import math
import random
import numpy as np
import scipy.misc
import skimage.color
import skimage.io
from PIL import Image
import cv2
import urllib.request
import shutil
import logging

import torch
from torch.utils.data import Dataset

from Configs.config import Config
from Utils import utils
from DataSets.imdb import IMDB

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


class CocoDataset(IMDB, Dataset):

    def __getitem__(self, image_index):
        image_id = self.image_ids[image_index]
        image, image_meta, gt_class_ids, gt_boxes = \
            self.load_image_gtbbox(image_id=image_id, config=self.config,
                                   rgbmean=True, augment=False, box_format='yxhw')

        # 去空值
        if not np.any(gt_class_ids > 0):
            return None

        # 去多余
        # If more instances than fits in the array, sub-sample from them.
        if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
            ids = np.random.choice(np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]

        # 转换为Tensor
        image = torch.from_numpy(image).float()
        image_meta = torch.from_numpy(image_meta)
        gt_class_ids = torch.from_numpy(gt_class_ids)
        gt_boxes = torch.from_numpy(gt_boxes).float()

        return image, image_meta, gt_class_ids, gt_boxes

    def __getitemx__(self, image_index):
        """获取单张image数据，区别于generate获取一个batch的images数据"""

        image_id = self.image_ids[image_index]

        image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
            self.load_image_gt(image_id=image_id, config=self.config, rgbmean=True, augment=False, chw=True)

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

        # 转换为Tensor
        image = torch.from_numpy(image).float()
        image_meta = torch.from_numpy(image_meta)
        gt_class_ids = torch.from_numpy(gt_class_ids)
        gt_boxes = torch.from_numpy(gt_boxes).float()
        gt_masks = torch.from_numpy(gt_masks).float()

        return image, image_meta, gt_class_ids, gt_boxes, gt_masks

    def __len__(self):
        return len(self.image_info)

    def __repr__(self):
        return self.__class__.__name__

    def load_data(self, *args, **kwargs):
        return self.load_coco(*args, **kwargs)

    def load_coco(self, data_dir, subset, year, class_ids=None, class_map=None, auto_download=None, **kwargs):
        """
        :param data_dir:
        :param subset:
        :param year:
        :param class_ids:  class ids in original dataset. == category ids. zhiding jiazai qizhon de naxie lei.
        :param class_map:
        :param auto_download:
        :param kwargs:
        :return:
        """

        if auto_download is True:
            self.auto_download(data_dir, subset, year)

        coco = COCO("{}/annotations/instances_{}{}.json".format(data_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(data_dir, subset, year)

        # Load all classes or a subset?
        # 添加需要的类id，所有类，还是其子集
        if not class_ids:
            # All classes, Note in coco2014, class index from 0 to 90, skip some val, so only have 80 classes.
            class_ids = sorted(coco.getCatIds())
        # Load  all images or a subset?
        # 添加需要的图片id，还是某些类构成的子集
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            # 82081  in coco2014  < coco.getImgIds(), others 783 not contain any  classes.
            image_ids = list(set(image_ids))
        else:
            # All images   # 82783  in coco2014
            image_ids = list(coco.imgs.keys())

        # 根据选择出的 class-ids 和 image-ids 添加更加详细的 class-info 和 image-info

        # Add classes 添加类信息 class_info
        for id in class_ids:
            # cat = coco.loadCats(id)      # [{'supercategory': 'person', 'id': 1, 'name': 'person'}]
            self.add_class("coco", id, coco.loadCats(id)[0]["name"])

        # Add images  添加图片信息 image_info
        for i in image_ids:
            self.add_image(
                image_source="coco",
                image_id=i,
                image_path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None)))

        if 'return_coco' in kwargs.keys() and kwargs['return_coco']:
            return coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        一张图片上可能有多个物体，且分属不同的类.
        image_id → (instance_mask，class_ID)×N

        不同数据集使用不同方法存储masks.
        本函数将其转换为统一格式:bitmap [height, width, instances]。

        Returns:
        masks: A bool array of shape [height, width, instance count] with one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        # 构造形为[height, weight, instance_count]的mask,且列出每个mask实例的class IDs
        for annotation in image_info["annotations"]:
            class_id = self.map_source_class_id("coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"], image_info["width"])
                # m: 2D numpy array [height, width] bool值
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                # 处理小物体标注
                if m.max() < 1:
                    print('\nmask pixels is less than 1 ...\n')
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
                        raise Exception('todo: mask is smaller than given dimensions!')
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def load_bbox(self, image_id, box_format='y1x1y2x2'):
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_bbox(image_id)

        instance_bboxes = []
        class_ids = []
        for annotation in image_info["annotations"]:
            class_id = self.map_source_class_id("coco.{}".format(annotation['category_id']))
            if class_id:
                x1, y1, w, h = annotation['bbox']
                if x1 < 0 or y1 < 0 or h < 1 or w < 1 or annotation['area'] < 0:
                    print('warning: bbox value error... todo???')
                    continue
                if box_format == 'y1x1y2x2':
                    # x2, y2 should not in bbox @ cocoapi/loadRes()
                    x2, y2 = x1 + w, y1 + h
                    instance_bboxes.append(np.array([y1, x1, y2, x2]))
                elif box_format == 'y1x1hw':
                    instance_bboxes.append(np.array([y1, x1, h, w]))
                elif box_format == 'yxhw':
                    ctr_y, ctr_x = y1 + 0.5*h, x1 + 0.5*w
                    instance_bboxes.append(np.array([ctr_y, ctr_x, h, w]))
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            bboxes = np.stack(instance_bboxes, axis=0)
            class_ids = np.array(class_ids, dtype=np.int32)
            return bboxes, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_bbox(image_id)

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

    def auto_download(self, dataDir, dataType, dataYear):
        pass
        return None

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    def build_results(self, image_ids, class_ids, scores, rois, masks):
        """Arrange resutls to match COCO specs in http://cocodataset.org/#format
            按COCO数据集的官方格式，重新整理检测结果 -> results中的每个ROIs与annotations格式相同
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
