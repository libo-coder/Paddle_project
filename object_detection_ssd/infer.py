# -*- coding: UTF-8 -*-
"""
使用训练好的模型开始预测。
    1.加载模型
    2.预测图片resize
    3.非极大值抑制（NMS是目标检测的后处理模块，主要用于删除高度冗余的bouding_box）
    4.绘制矩形框
@author: libo
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import time
import paddle.fluid as fluid

from PIL import Image

from PIL import ImageDraw

target_size = [3, 300, 300]
nms_threshold = 0.45              #非极大值抑制：NMS是目标检测的后处理模块，主要用于删除高度冗余的bouding_box
confs_threshold = 0.5

#创建预测用的exe
place = fluid.CPUPlace()
exe = fluid.Executor(place)
path = "./ssd-model-back"

#从指定路径加载模型
[inference_program, feed_target_names, fetch_targets] = \
    fluid.io.load_inference_model(dirname=path,
                                  params_filename='mobilenet-ssd-final-params',
                                  model_filename='mobilenet-ssd-final-model',
                                  executor=exe)
print(fetch_targets)


def draw_bbox_image(img, nms_out, save_name):
    """
    给图片画上外接矩形框
    :param img:
    :param nms_out:
    :param save_name:
    :return:
    """
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)
    for dt in nms_out:
        if dt[1] < confs_threshold:
            continue
        category_id = dt[0]
        bbox = dt[2:]
        #根据网络输出，获取矩形框的左上角、右下角坐标相对位置
        xmin, ymin, xmax, ymax = clip_bbox(dt[2:])
        draw.rectangle((xmin * img_width, ymin * img_height, xmax * img_width, ymax * img_height), None, 'red')
    img.save(save_name)


def clip_bbox(bbox):
    """
    截断矩形框
    :param bbox:
    :return:
    """
    xmin = max(min(bbox[0], 1.), 0.)
    ymin = max(min(bbox[1], 1.), 0.)
    xmax = max(min(bbox[2], 1.), 0.)
    ymax = max(min(bbox[3], 1.), 0.)
    return xmin, ymin, xmax, ymax


def resize_img(img, target_size):
    """
    保持比例的缩放图片
    :param img:
    :param target_size:
    :return:
    """
    percent_h = float(target_size[1]) / img.size[1]
    percent_w = float(target_size[2]) / img.size[0]
    percent = min(percent_h, percent_w)
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    w_off = (target_size[1] - resized_width) / 2
    h_off = (target_size[2] - resized_height) / 2
    img = img.resize((target_size[1], target_size[2]), Image.ANTIALIAS)
    return img


def read_image(img_path):
    """
    读取图片
    :param img_path:
    :return:
    """
    img = Image.open(img_path)
    resized_img = img.copy()
    img = resize_img(img, target_size)
    if img.mode != 'RGB':                                       #颜色通道为RGB
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1))  #转置 HWC to CHW 数据通道
    img -= 127.5                                                #
    img *= 0.007843                                             #归一化到-1到1
    img = img[np.newaxis, :]
    return img, resized_img


def infer(image_path):
    """
    预测，将结果保存到一副新的图片中
    :param image_path:
    :return:
    """
    #将预测图片按比例进行缩放
    tensor_img, resized_img = read_image(image_path)
    t1 = time.time()
    #执行预测，并获取预测结果
    nmsed_out = exe.run(inference_program,
                        feed={feed_target_names[0]: tensor_img},
                        fetch_list=fetch_targets,
                        return_numpy=False)
    period = time.time() - t1
    print("predict result:{0} cost time:{1}".format(nmsed_out, "%2.2f sec" % period))
    nmsed_out = np.array(nmsed_out[0])        #进行非极大值抑制
    last_dot_index = image_path.rfind('.')
    out_path = image_path[:last_dot_index]
    out_path += '-reslut.jpg'
    print("result save to:", out_path)
    #在图片上绘制矩形框
    draw_bbox_image(resized_img, nmsed_out, out_path)


#开始推测
image_path = 'dog-cat.jpg'
infer(image_path)