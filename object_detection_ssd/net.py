# coding=utf-8
"""
定义基于 mobile-net 的 SSD 网络结构
mobile-net为移动端和嵌入式端深度学习应用设计的网络，使得在cpu上也能达到理想的速度要求。
@author: libo
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import uuid
import numpy as np
import time
import six
import math
import paddle
import paddle.fluid as fluid
import logging
import xml.etree.ElementTree
import codecs

from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from PIL import Image, ImageEnhance, ImageDraw

class MobileNetSSD:
    def __init__(self):
        pass

    def conv_bn(self,
                input,
                filter_size,
                num_filters,
                stride,
                padding,
                num_groups=1,
                act='relu',
                use_cudnn=True):
        parameter_attr = ParamAttr(learning_rate=0.1, initializer=MSRA())
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=parameter_attr,
            bias_attr=False)
        return fluid.layers.batch_norm(input=conv, act=act)

    def depthwise_separable(self, input, num_filters1, num_filters2, num_groups, stride, scale):
        depthwise_conv = self.conv_bn(
            input=input,
            filter_size=3,
            num_filters=int(num_filters1 * scale),
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            use_cudnn=False)

        pointwise_conv = self.conv_bn(
            input=depthwise_conv,
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0)
        return pointwise_conv

    def extra_block(self, input, num_filters1, num_filters2, num_groups, stride, scale):
        # 1x1 conv
        pointwise_conv = self.conv_bn(
            input=input,
            filter_size=1,
            num_filters=int(num_filters1 * scale),
            stride=1,
            num_groups=int(num_groups * scale),
            padding=0)

        # 3x3 conv
        normal_conv = self.conv_bn(
            input=pointwise_conv,
            filter_size=3,
            num_filters=int(num_filters2 * scale),
            stride=2,
            num_groups=int(num_groups * scale),
            padding=1)
        return normal_conv

    def net(self, num_classes, img, img_shape, scale=1.0):
        # 300x300
        tmp = self.conv_bn(img, 3, int(32 * scale), 2, 1)
        # 150x150
        tmp = self.depthwise_separable(tmp, 32, 64, 32, 1, scale)
        tmp = self.depthwise_separable(tmp, 64, 128, 64, 2, scale)
        # 75x75
        tmp = self.depthwise_separable(tmp, 128, 128, 128, 1, scale)
        tmp = self.depthwise_separable(tmp, 128, 256, 128, 2, scale)
        # 38x38
        tmp = self.depthwise_separable(tmp, 256, 256, 256, 1, scale)
        tmp = self.depthwise_separable(tmp, 256, 512, 256, 2, scale)

        # 19x19
        for i in range(5):
            tmp = self.depthwise_separable(tmp, 512, 512, 512, 1, scale)
        module11 = tmp
        tmp = self.depthwise_separable(tmp, 512, 1024, 512, 2, scale)

        # 10x10
        module13 = self.depthwise_separable(tmp, 1024, 1024, 1024, 1, scale)
        module14 = self.extra_block(module13, 256, 512, 1, 2, scale)
        # 5x5
        module15 = self.extra_block(module14, 128, 256, 1, 2, scale)
        # 3x3
        module16 = self.extra_block(module15, 128, 256, 1, 2, scale)
        # 2x2
        module17 = self.extra_block(module16, 64, 128, 1, 2, scale)
        #生成SSD算法的候选框。从多个特征图中，进行预测分类边界框。
        mbox_locs, mbox_confs, box, box_var = fluid.layers.multi_box_head(
            inputs=[module11, module13, module14, module15, module16, module17],        # 输入变量列表
            image=img,                                                                  # 输入图像数据
            num_classes=num_classes,                                                    # 类的数量
            min_ratio=20,                                                               # 生成候选框的最小比例
            max_ratio=90,                                                               # 生成候选框的最大比例
            aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2., 3.], [2., 3.]],     # 生成候选框的宽高比
            base_size=img_shape[2],                                                     # 300
            offset=0.5,                                                                 # 候选框中心偏移
            flip=True)                                                                  # 是否翻转宽高比

        return mbox_locs, mbox_confs, box, box_var   # 返回gound_truth的位置（中心点的坐标、长、宽）、预测框对输入的置信度、候选框、方差


