#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/4/3 18:53'
__author__ = 'ooo'

import torch
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import BasicBlock, Bottleneck, model_urls
from torch import nn
import math


class ResNet(nn.Module):
    """
    修改自 torchvision.models.resnet.ResNet
    返回多个层级的特征图[C1, C2, C3, C4, C5], 用于构造FPN
    """
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        C1 = x
        x = self.layer1(x)
        C2 = x
        x = self.layer2(x)
        C3 = x
        x = self.layer3(x)
        C4 = x
        x = self.layer4(x)
        C5 = x
        # 在检测算法中不需要计算全连接层的输出x
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return [C1, C2, C3, C4, C5]


def resnet18(pretrained=False, model_dir=None, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir))
    return model


def resnet34(pretrained=False, model_dir=None, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir))
    return model


def resnet50(pretrained=False, model_dir=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir))
    return model


def resnet101(pretrained=False, model_dir=None, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir))
    return model


def resnet152(pretrained=False, model_dir=None, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir))
    return model


def resnet(arch, pretrained=False, model_dir=None, include=None):
    """
    :param arch:
    :param pretrained:
    :param include: include layers
    :return:
    """
    arch = arch.lower()

    if arch == 'resnet18':
        model = resnet18(pretrained, model_dir)
    elif arch == 'resnet34':
        model = resnet34(pretrained, model_dir)
    elif arch == 'resnet50':
        model = resnet50(pretrained, model_dir)
    elif arch == 'resnet101':
        model = resnet101(pretrained, model_dir)
    elif arch == 'resnet152':
        model = resnet152(pretrained, model_dir)
    else:
        raise ValueError('错误的arch代码！')

    # error: OrderedDict can't be changed during iteration!!
    # if exclude is not None:
    #     for name, child in model.named_children():
    #         if name in exclude:
    #             delattr(model, name)
    # [delattr(model, name) for name, _ in model.named_children() if name in exclude]
    if include is not None:
        new_model = torch.nn.Sequential()
        [new_model.add_module(n, m) for n, m in model.named_children() if n in include]
        model = new_model
    return model


def vgg(arch, pretrained=False, model_dir=None, include=None):
    pass


def desnet(arch, pretrained=False, model_dir=None, include=None):
    pass


def backbone(arch, pretrained=False, model_dir=None, include=None):
    if 'resnet' in arch:
        return resnet(arch, pretrained, model_dir, include)
    elif 'vgg' in arch:
        return vgg(arch, pretrained, model_dir, include)
    elif 'desnet' in arch:
        return desnet(arch, pretrained, model_dir, include)
    else:
        raise ValueError('Unknown Backbone Model: %s' % arch)
