#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/4/3 18:53'
__author__ = 'ooo'

import os
import math
import datetime
import re
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim

from Models.backbone import backbone
from Models.fpn import FPN
from Models.lscnet import LSCNet
from Layers.simple_net import SimpleNet
from Layers.anchor_proposal_rois import HotAnchorLayer, HotProposalLayer, RoiTargetLayer, RoiTransformLayer
from Layers.class_bbox_mask import ClassBoxNet, MaskNet
from Layers.detections import detection_layer, pyramid_detection_layer
from Layers.loss import compute_losses
from Utils import utils, visualize
from DataSets.coco_dataset import CocoDataset


class HotRCNN(nn.Module):
    def __init__(self, config):
        super(HotRCNN, self).__init__()
        self.config = config
        self.exp_dir = config.EXP_DIR
        self.set_log_dir()
        self.build(config=config)
        self.init_weights()
        self.train_loss_history = []
        self.val_loss_history = []

    def build(self, config):

        # 验证输入尺寸
        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")
        # 特征生成网络
        self.backbone = backbone(arch=config.BACKBONE_ARCH, pretrained=config.BACKBONE_PreTRAINED,
                                 model_dir=config.BACKBONE_DIR, include=config.BACKBONE_INCLUDE)

        # 特征融合网络
        if config.FEATURE_FUSION_METHOD == 'fpn':
            self.fpnet = FPN(channels=256)
        elif config.FEATURE_FUSION_METHOD == 'lsc':
            self.lscnet = LSCNet(indepth=config.LSC_IN_CHANNELS, outdepth=config.LSC_OUT_CHANNELS,
                                 kernel=config.LSC_KERNEL_SIZE, levels=1)
        elif config.FEATURE_FUSION_METHOD == 'simple':
            self.simplenet = SimpleNet(indepth=config.LSC_IN_CHANNELS, outdepth=config.LSC_OUT_CHANNELS,
                                       kernel=config.LSC_KERNEL_SIZE, levels=1)
        elif config.FEATURE_FUSION_METHOD == 'none':
            pass
        else:
            raise ValueError('未知的参数设定！FEATURE_FUSION_METHOD！')

        # Anchors生成层
        self.anchors_generate = HotAnchorLayer(scales=config.ANCHOR_SCALES,
                                               ratios=config.ANCHOR_ASPECTS,
                                               counts=config.ANCHORS_PER_IMAGE,
                                               level_nums=config.ANCHOR_LEVEL_NUMS,
                                               heat_method=config.ANCHOR_HEAT_METHOD,
                                               zero_area=config.ANCHOR_ZERO_AREA,
                                               image_shape=config.IMAGE_SHAPE)

        # Proposals选择层
        self.proposals_select = HotProposalLayer(mode='',
                                                 counts=config.PROPOSALS_PER_IMAGE,
                                                 image_shape=config.IMAGE_SHAPE,
                                                 level_nums=config.ANCHOR_LEVEL_NUMS)

        # ROIs-GT匹配层，Proposal-GTbox匹配，RoiTargetLayer, 训练阶段
        self.rois_target_match = RoiTargetLayer(config=config)

        # ROIs变换层，Pooling, Align, Wrap
        # self.rois_transform = RoiTransformLayer(config=config)

        # class bbox head
        self.class_bbox_net = ClassBoxNet(indepth=config.FEATURE_FUSION_CHANNELS,
                                          pool_size=config.BBOX_POOL_SIZE,
                                          image_shape=config.IMAGE_SHAPE,
                                          fmap_stride=config.FEATURE_FUSION_STRIDES,
                                          class_nums=config.CLASSES_NUMS,
                                          level_nums=config.ANCHOR_LEVEL_NUMS)

        # mask head
        self.mask_net = MaskNet(indepth=config.FEATURE_FUSION_CHANNELS,
                                pool_size=config.MASK_POOL_SIZE,
                                image_shape=config.IMAGE_SHAPE,
                                fmap_stride=config.FEATURE_FUSION_STRIDES,
                                class_nums=config.CLASSES_NUMS,
                                level_nums=config.ANCHOR_LEVEL_NUMS)

        # detection results
        self.detection_layer = pyramid_detection_layer

        # compute loss
        self.compute_losses = compute_losses

        # Fix batch norm layers
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        self.apply(set_bn_fix)

    def predict(self, inputs, mode):
        """
        0.输入较正
        1.特征计算
        2.特征融合
        3.锚点生成
        4.建议区选择
        5.兴趣区变换 rois，rois+gt
        6.头部网络
        7.返回结果
        forward, tain_batch, val_batch
        :return:
        """
        molded_images, image_metas = inputs[0], inputs[1]

        if mode == 'training':
            self.train()
            self.apply(self.frozen_batch_normal)
        elif mode == 'inference':
            self.eval()

        fmaps = self.backbone(molded_images)

        if self.config.FEATURE_FUSION_METHOD == 'fpn':
            fmaps = self.fpnet(fmaps)
        elif self.config.FEATURE_FUSION_METHOD == 'lsc':
            fmaps = self.lscnet(fmaps)
        elif self.config.FEATURE_FUSION_METHOD == 'none':
            fmaps = [fmaps]
        else:
            raise ValueError('unknown method!')

        anchors = self.anchors_generate(fmaps)

        proposals = self.proposals_select(anchors, fmaps)

        if mode == 'training':
            gt_class_ids = inputs[2]
            gt_boxes = inputs[3]
            gt_masks = inputs[4]

            # 归一化坐标
            h, w = self.config.IMAGE_SHAPE[0:2]
            scale = Variable(torch.FloatTensor(np.array([h, w, h, w]).astype(np.int32)), requires_grad=False).cuda()
            gt_boxes = gt_boxes / scale

            # 生成检测目标 zero-padded
            rois, target_class_ids, target_deltas, target_masks = \
                self.rois_target_match(proposals, gt_class_ids, gt_boxes, gt_masks)

            if rois.size(0):
                # fmaps & rois 的 level_nums 相对应
                class_logits, class_probs, pred_deltas = self.class_bbox_net(fmaps, rois)
                pred_masks = self.mask_net(fmaps, rois)
            else:
                class_logits = Variable(torch.FloatTensor().cuda())
                class_probs = Variable(torch.FloatTensor().cuda())
                pred_deltas = Variable(torch.FloatTensor().cuda())
                pred_masks = Variable(torch.FloatTensor().cuda())

            # target_class_ids: [b, N, (class_id)]      class_logits: [b*N, class_nums, (logits)]
            # target_deltas: [b, N, (dy, dx, dw, dh)]   pred_deltas: [b*N, class_nums, (dy, dx, dw, dh)]
            # target_masks: [b, N, h, w]                pred_masks: [b*N, class_nums, h', w']
            return [target_class_ids, class_logits, target_deltas, pred_deltas, target_masks, pred_masks]

        elif mode == 'inference':

            # proposals ：[[batch, N, (y1, x1, y2, x2), ., ...]]
            rois = proposals[0]

            class_logits, class_probs, pred_deltas = self.class_bbox_net(fmaps, rois)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
            detections, allboxes = self.detection_layer(self.config, rois, class_probs, pred_deltas, image_metas, True)

            # Create masks for detections
            pred_masks = self.mask_net(fmaps, allboxes)
            pred_masks = pred_masks.unsqueeze(0)

            return [detections, pred_masks]

    def detect(self, images):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Convert images to torch tensor
        molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float()

        # To GPU
        if self.config.GPU_COUNT:
            molded_images = molded_images.cuda()

        # Wrap in variable
        molded_images = Variable(molded_images, volatile=True)

        # Run object detection
        detections, pred_masks = self.predict([molded_images, image_metas], mode='inference')

        # Convert to numpy
        detections = detections.data.cpu().numpy()
        pred_masks = pred_masks.permute(0, 1, 3, 4, 2).data.cpu().numpy()  # todo [b, N, H, W, C] ?

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks = \
                self.unmold_detections(detections[i], pred_masks[i],
                                       image.shape, windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def train_model(self, train_dataset, val_dataset, learning_rate, epochs, layers):
        """
        1.读取数据   2.遍历epoch  3.构造优化器
        :param train_dataset:
        :param val_dataset:
        :param learning_rate:
        :param epochs:
        :param layers:
        :return:
        """
        # Train
        utils.log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        utils.log("Checkpoint will be saved at: {}\n".format(self.checkpoint_path))

        self.set_trainable(layers)

        # data iterator # not generator!
        trainset_iter = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
        valset_iter = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)

        # Optimizer object
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        trainables_wo_bn = [param for name, param in self.named_parameters() if param.requires_grad and 'bn' not in name]
        trainables_only_bn = [param for name, param in self.named_parameters() if param.requires_grad and 'bn' in name]
        optimizer = optim.SGD([
            {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
            {'params': trainables_only_bn}
        ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)

        for epoch in range(self.epoch, epochs+1):
            utils.log("Epoch {}/{}.".format(epoch, epochs))

            # Training
            train_loss = self.train_epoch(trainset_iter, optimizer, self.config.TRAIN_STEPS_PER_EPOCH)

            # Validation
            val_loss = self.valid_epoch(valset_iter, self.config.VAL_STEPS_PER_EPOCH)

            # Statistics
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            visualize.plot_loss(self.train_loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)

            # Save model
            torch.save(self.state_dict(), self.checkpoint_path.format(epoch=epoch))

        if self.epoch < epochs + 1:
            self.epoch = epochs + 1
        else:
            utils.log("Train has been done over in current stage !!!")

    def train_epoch(self, dataiterator, optimizer, steps):
        batch_count = 0
        loss_sum = 0
        step = 0

        for inputs in dataiterator:
            batch_count += 1

            images = inputs[0]
            image_metas = inputs[1]
            gt_class_ids = inputs[2]
            gt_boxes = inputs[3]
            gt_masks = inputs[4]

            # image_metas as numpy array
            image_metas = image_metas.numpy()

            # Wrap in variables
            images = Variable(images)
            gt_class_ids = Variable(gt_class_ids)
            gt_boxes = Variable(gt_boxes)
            gt_masks = Variable(gt_masks)

            # To GPU
            if self.config.GPU_COUNT:
                images = images.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()
                gt_masks = gt_masks.cuda()

            # Run object detection
            target_class_ids, class_logits, target_deltas, pred_deltas, target_masks, pred_masks = \
                self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')

            # Compute losses
            class_loss, bbox_loss, mask_loss = self.compute_losses(target_class_ids, class_logits,
                                                                   target_deltas, pred_deltas,
                                                                   target_masks, pred_masks)
            loss = class_loss + bbox_loss + mask_loss

            # Backpropagation
            if (batch_count % self.config.BATCH_SIZE) == 0:
                optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.parameters(), 5.0)
            if (batch_count % self.config.BATCH_SIZE) == 0:
                optimizer.step()
                batch_count = 0

            # Progress
            utils.printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                   suffix="Complete - loss: {:.5f} - class_loss: {:.5f} - bbox_loss: {:.5f} - mask_loss: {:.5f}".format(
                                       loss.data.cpu()[0], class_loss.data.cpu()[0], bbox_loss.data.cpu()[0],
                                       mask_loss.data.cpu()[0]), length=10)

            # Statistics
            loss_sum += loss.data.cpu()[0] / steps

            # Break after 'steps' steps
            if step == steps - 1:
                break
            step += 1

        return loss_sum

    def valid_epoch(self, dataiterator, steps):

        step = 0
        loss_sum = 0

        for inputs in dataiterator:
            images = inputs[0]
            image_metas = inputs[1]
            gt_class_ids = inputs[2]
            gt_boxes = inputs[3]
            gt_masks = inputs[4]

            # image_metas as numpy array
            image_metas = image_metas.numpy()

            # Wrap in variables
            images = Variable(images, volatile=True)
            gt_class_ids = Variable(gt_class_ids, volatile=True)
            gt_boxes = Variable(gt_boxes, volatile=True)
            gt_masks = Variable(gt_masks, volatile=True)

            # To GPU
            if self.config.GPU_COUNT:
                images = images.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()
                gt_masks = gt_masks.cuda()

            # Run object detection
            target_class_ids, class_logits, target_deltas, pred_deltas, target_masks, pred_masks = \
                self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')

            if not target_class_ids.size():
                continue

            # Compute losses
            class_loss, bbox_loss, mask_loss = self.compute_losses(target_class_ids, class_logits,
                                                                   target_deltas, pred_deltas,
                                                                   target_masks, pred_masks)
            loss = class_loss + bbox_loss + mask_loss

            # Progress
            utils.printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                   suffix="Complete - loss: {:.5f}  - class_loss: {:.5f} - bbox_loss: {:.5f} - mask_loss: {:.5f}".format(
                                       loss.data.cpu()[0], class_loss.data.cpu()[0], bbox_loss.data.cpu()[0],
                                       mask_loss.data.cpu()[0]), length=10)

            # Statistics
            loss_sum += loss.data.cpu()[0] / steps
            # Break after 'steps' steps
            if step == steps - 1:
                break
            step += 1

        return loss_sum

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def load_weights(self, filepath, source):
        if source in ['hrcnn', 'ckpt']:
            if os.path.exists(filepath):
                self.load_state_dict(torch.load(filepath))
            else:
                raise ValueError('未找到权值文件！')
        elif source == 'backbone':
            state_dict = torch.load(filepath)
            own_state = self.state_dict()
            for name, param in state_dict.items():
                name = 'backbone.' + name
                if name in own_state:
                    if isinstance(param, nn.Parameter):
                        # backwards compatibility for serialized parameters
                        param = param.data
                    try:
                        own_state[name].copy_(param)
                    except Exception:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
        else:
            raise ValueError('Unknow Weights Source: %s !' % source)

        # Update the log directory
        self.set_log_dir(filepath)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def frozen_weights(self, layer_regex):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        for name, param in self.named_parameters():
            trainable = bool(re.fullmatch(layer_regex, name))
            if not trainable:
                param[1].requires_grad = False

    @staticmethod
    def frozen_batch_normal(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def set_trainable(self, layers, layer_regex=None):
        """
        指定需要训练的层
        """
        # Pre-defined layer regular expressions
        if self.config.FEATURE_FUSION_METHOD == 'fpn':
            default_layer_regex = {
                # all layers but the backbone
                "heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|"
                         r"(class_bbox.*)|(mask.*)",
                # From a specific Resnet stage and up
                "2+": r"(resnet.layer2.*)|(resnet.layer3.*)|(resnet.layer4.*)|"
                      r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|"
                      r"(class_bbox.*)|(mask.*)",
                "3+": r"(resnet.layer3.*)|(resnet.layer4.*)|"
                      r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|"
                      r"(class_bbox.*)|(mask.*)",
                "4+": r"(resnet.layer4.*)|"
                      r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|"
                      r"(class_bbox.*)|(mask.*)",

                # All layers
                "all": ".*",
            }
        elif self.config.FEATURE_FUSION_METHOD in ['lsc', 'none']:
            default_layer_regex = {
                # all layers but the backbone
                "heads": r"(lscnet.*)|(class_bbox.*)|(mask.*)",
                # From a specific Resnet stage and up
                "2+": r"(resnet.layer2.*)|(resnet.layer3.*)|(resnet.layer4.*)|"
                      r"(lscnet.*)|(class_bbox.*)|(mask.*)",
                "3+": r"(resnet.layer3.*)|(resnet.layer4.*)|(lscnet.*)|(class_bbox.*)|(mask.*)",
                "4+": r"(resnet.layer4.*)|(lscnet.*)|(class_bbox.*)|(mask.*)",
                # All layers
                "all": ".*",
            }
        else:
            raise ValueError('未知的训练层设定！')

        layer_regex = layer_regex if layer_regex else default_layer_regex
        if layers in layer_regex.keys():
            layers = layer_regex[layers]
        else:
            raise KeyError('字典键 %s 不存在！' % layers)

        # self.frozen_weights(layers)
        for name, param in self.named_parameters():
            trainable = bool(re.fullmatch(layers, name))
            if not trainable:
                param.requires_grad = False
            else:
                pass
                # print('%s will be training!' % name)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = CocoDataset.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            molded_image = CocoDataset.mold_image(molded_image, self.config.MEAN_PIXEL)
            # Build image_meta
            image_meta = CocoDataset.compose_image_meta(
                0, image.shape, window,
                np.zeros([self.config.CLASSES_NUMS], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask: [N, height, width, num_classes]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

        # Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) \
            if full_masks else np.empty((0,) + masks.shape[1:3])

        return boxes, class_ids, scores, full_masks

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.
        路径顺序是：ProjectDir / model_dir / log_dir / checkpoint_path
        工程目录 root_dir / 模型目录 model_dir / 实验目录 exp-log-dir /日志ckpt文件 check_point-model

        log_dir下有2种文件：checkpoint.h5（模型） ，events.os（日志）

        model_path： 半成品的ckpt模型 || 完整的hrcnn模型 || resnet模型
        regex_match： 只有匹配为ckpt模型时，才会自动推算 self.log_dir, self.epoch

        在ckpt情况下，使用从该ckpt文件中解码出的 self.log_dir, self.epoch.
        在后2种情况下，使用基于当期日期和config.NAME组合出的 self.log_dir, 以及sel.epoch=0

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 1  # 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        # model_path 可能是 find_last() 返回的checkpoint的路径
        # 可以从checkpoint的文件名中解码出最后一次训练的epoch值。
        # 基于正则匹配，只有当其是ckpt的路径时，才会进行查找匹配。
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            # /config.NAME + 年月日时分 / hot_rcnn_xxx.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/hot\_rcnn\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) + 1  # m.group(6): 匹配到的最后一个d{4}

        # Directory for training logs 根据解码出的日期，或当前日期，拼写出日志ckpt目录
        self.log_dir = os.path.join(self.exp_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "hot_rcnn_{}_*epoch*.h5".format(self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

        return self.epoch, self.log_dir, self.checkpoint_path

    def get_backone_path(self, path=None):
        if path is not None:
            return path
        return self.config.BACKBONE_PATH

    def get_weights_path(self, dataset, path=None):
        """Downloads ImageNet trained weights from Keras.
        ImageNet上的模型权值，HotRCNN模型权值
        Returns path to weights file.
        """
        assert dataset in ['imagenet', 'coco14', 'coco17', 'voc07', 'voc11'], '数据集不匹配'
        if dataset == 'imagenet':
            TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/' \
                                     'releases/download/v0.2/' \
                                     'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
            # from keras.utils.data_utils import get_file
            # weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            #                         TF_WEIGHTS_PATH_NO_TOP,
            #                         cache_subdir='models',
            #                         md5_hash='a268eb855778b3df3c7506639542a6af')
            weights_path = self.config.RESNET_MODEL_PATH
            return weights_path

        elif dataset == 'coco14':
            weights_path = None
            return weights_path

        elif dataset == 'coco17':
            weights_path = None
            return weights_path

        elif dataset == 'hrcnn':
            hrcnn_model_path = path if path else self.config.HRCNN_MODEL_PATH
            return hrcnn_model_path

        else:
            raise ValueError('未知的dataset参数！')

    def get_ckpt_path(self, log_dir=None, ckpt_name=None):
        """Finds the last checkpoint file of the last trained model in the
        model directory.

        文件位置：/sef.exp_dir/log_dir/ckpt.h5

        即可通过查找训练历史文件，自动推算出最后一次训练的ckpt.h5文件
        也可通过log_dir和ckpt_name直接指定要加载的ckpt.h5文件

        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        if not log_dir:
            # Get directory names. Each directory corresponds to a model
            dir_names = next(os.walk(self.exp_dir))[1]
            key = self.config.NAME.lower()
            dir_names = filter(lambda f: f.startswith(key), dir_names)
            dir_names = sorted(dir_names)
            if not dir_names:
                return None, None
            # Pick last directory
            dir_name = os.path.join(self.exp_dir, dir_names[-1])
        else:
            key = self.config.NAME.lower()
            assert log_dir.startswith(key), '日志文件夹log_dir，必须以config.NAME.lower()开头！'
            dir_name = log_dir

        if not ckpt_name:
            # Find the last checkpoint
            checkpoints = next(os.walk(dir_name))[2]
            checkpoints = filter(lambda f: f.startswith("hot_rcnn"), checkpoints)
            checkpoints = sorted(checkpoints)
            if not checkpoints:
                return dir_name, None
            checkpoint = os.path.join(dir_name, checkpoints[-1])
        else:
            checkpoint = os.path.join(dir_name, ckpt_name)
        print('\n查找到最近训练节点: {}\n'.format(checkpoint))
        return dir_name, checkpoint

    def __repr__(self):
        return self.__class__.__name__
