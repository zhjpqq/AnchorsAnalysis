#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2018/4/3 18:53'
__author__ = 'ooo'

import torch
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, model_urls


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





