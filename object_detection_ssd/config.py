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

logger = None
train_parameters = {
    "input_size": [3, 300, 300],  # 图片的维度
    "class_dim": -1,              # 分类数
    "label_dict": {},             # 存放标签字典
    "image_count": -1,            # 训练图片数量
    "log_feed_image": False,
    "pretrained": True,           # 是否使用预训练的模型
    "continue_train": True,
    # "pretrained_model_dir": "./pretrained-model",       # 预训练的 mobilenet 模型存放路径
    "pretrained_model_dir": "E:/project/dataset/paddle_ssd/data7948/pretrained-model",       # 预训练的 mobilenet 模型存放路径
    "save_model_dir": "./ssd-model-back",                    # 训练后的模型保存路径
    "model_prefix": "mobilenet-ssd",                    # 模型路径前缀
    "data_dir": "E:/project/dataset/paddle_ssd/data4379/pascalvoc",     # 数据集解压后存放的目录
    "mean_rgb": [127.5, 127.5, 127.5],              # 常用图片的三通道均值，通常来说需要先对训练数据做统计，此处仅取中间值
    "file_list": "train.txt",                       # 存放训练集图片和标注文件的对应关系
    "mode": "train",                                # train 或者 test
    "multi_data_reader_count": 1,
    "num_epochs": 1,                                # 训练轮数
    "train_batch_size": 16,                         # 训练集 batch_size 大小
    "use_gpu": False,                                # 是否使用 gpu, True
    "apply_distort": True,
    "apply_expand": True,
    "apply_corp": True,
    "image_distort_strategy": {                     # 图像增强的一堆参数
        "expand_prob": 0.5,
        "expand_max_ratio": 4,
        "hue_prob": 0.5,
        "hue_delta": 18,
        "contrast_prob": 0.5,
        "contrast_delta": 0.5,
        "saturation_prob": 0.5,
        "saturation_delta": 0.5,
        "brightness_prob": 0.5,
        "brightness_delta": 0.125
    },
    "rsm_strategy": {                                # 一种自适应学习率的方法
        "learning_rate": 0.001,
        "lr_epochs": [40, 60, 80, 100],
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01],
    },
    "momentum_strategy": {                           # 暂未使用
        "learning_rate": 0.1,
        "decay_steps": 2 ** 7,
        "decay_rate": 0.8
    },
    "early_stop": {
        "sample_frequency": 50,
        "successive_limit": 3,
        "min_loss": 1.28,                           # 最小的损失
        "min_curr_map": 0.86                        # 最小的 mAP 值
    }
}


# 初始化 train_parameters 中的参数
def init_train_parameters():
    file_list = os.path.join(train_parameters['data_dir'], "train.txt")
    label_list = os.path.join(train_parameters['data_dir'], "label_list")
    index = 0
    with codecs.open(label_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        for line in lines:
            train_parameters['label_dict'][line.strip()] = index
            index += 1
        train_parameters['class_dim'] = index
    with codecs.open(file_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        train_parameters['image_count'] = len(lines)