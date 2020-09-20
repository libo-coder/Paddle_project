# coding=utf-8
"""
自定义用户数据读取器。因为图像处理比较多，批处理时会很慢，可能导致数据处理时间比真实计算模型的时间还要长！为了尽量避免这种情况，训练时使用并行化的数据读取器。
同时，为了方便训练中能够验证当前的效果，中间验证的时候使用同步数据读取器
原本验证的数据不应该和训练数据混着用，此处仅仅为了示例，真实训练，建议将两批数据分开
@author: libo
"""
import os
import numpy as np
from PIL import Image
from config import train_parameters
import xml.etree.ElementTree
import math
import paddle
import paddle.fluid as fluid

from dataset import preprocess


def custom_reader(file_list, data_dir, mode):
    def reader():
        np.random.shuffle(file_list)
        for line in file_list:
            if mode == 'train' or mode == 'eval':
                image_path, label_path = line.split()
                image_path = os.path.join(data_dir, image_path)
                label_path = os.path.join(data_dir, label_path)
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                im_width, im_height = img.size
                # layout: label | xmin | ymin | xmax | ymax | difficult
                bbox_labels = []
                root = xml.etree.ElementTree.parse(label_path).getroot()
                for object in root.findall('object'):
                    bbox_sample = []
                    bbox_sample.append(float(train_parameters['label_dict'][object.find('name').text]))
                    bbox = object.find('bndbox')
                    difficult = float(object.find('difficult').text)
                    bbox_sample.append(float(bbox.find('xmin').text) / im_width)
                    bbox_sample.append(float(bbox.find('ymin').text) / im_height)
                    bbox_sample.append(float(bbox.find('xmax').text) / im_width)
                    bbox_sample.append(float(bbox.find('ymax').text) / im_height)
                    bbox_sample.append(difficult)
                    bbox_labels.append(bbox_sample)
                img, sample_labels = preprocess(img, bbox_labels, mode)
                sample_labels = np.array(sample_labels)
                if len(sample_labels) == 0:
                    continue
                boxes = sample_labels[:, 1:5]
                lbls = sample_labels[:, 0].astype('int32')
                difficults = sample_labels[:, -1].astype('int32')
                yield img, boxes, lbls, difficults
            elif mode == 'test':
                img_path = os.path.join(data_dir, line)
                yield Image.open(img_path)

    return reader


# 多进程 reader 使用 python 多进程从 reader 中读取数据
def multi_process_custom_reader(file_path, data_dir, num_workers, mode):
    file_path = os.path.join(data_dir, file_path)
    readers = []
    images = [line.strip() for line in open(file_path)]
    n = int(math.ceil(len(images) // num_workers))
    image_lists = [images[i: i + n] for i in range(0, len(images), n)]
    for l in image_lists:
        readers.append(paddle.batch(custom_reader(l, data_dir, mode),
                                    batch_size=train_parameters['train_batch_size'],
                                    drop_last=True))
    return paddle.reader.multiprocess_reader(readers, False)


def create_eval_reader(file_path, data_dir, mode):
    file_path = os.path.join(data_dir, file_path)
    images = [line.strip() for line in open(file_path)]
    return paddle.batch(custom_reader(images, data_dir, mode),
                        batch_size=train_parameters['train_batch_size'],
                        drop_last=True)