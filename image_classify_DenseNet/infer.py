# coding=utf-8
"""
模型预测
@author: libo
"""
import paddle.fluid as fluid
import paddle
import reader
from net import DenseNet
import numpy as np
import os
from config import train_parameters, init_train_parameters
from PIL import Image


def resize_img(img, target_size):
    """
    强制缩放图片
    :param img:
    :param target_size:
    :return:
    """
    img = img.resize((target_size[1], target_size[2]), Image.BILINEAR)
    return img

def read_img(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = resize_img(img, train_parameters['input_size'])
    # HWC--->CHW && normalized
    img = np.array(img).astype('float32')
    img -= train_parameters['mean_rgb']
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img *= 0.007843  # 像素值归一化
    img = np.array([img]).astype('float32')
    return img
    
def infer():
    with fluid.dygraph.guard():
        net = DenseNet("densenet", layers = 121, dropout_prob = train_parameters['dropout_prob'], class_dim = train_parameters['class_dim'])
        # load checkpoint
        model_dict, _ = fluid.dygraph.load_dygraph(train_parameters["save_persistable_dir"])
        net.load_dict(model_dict)
        print("checkpoint loaded")

        # start evaluate mode
        net.eval()
        
        label_dic = train_parameters["label_dict"]
        label_dic = {v: k for k, v in label_dic.items()}
        
        img_path = train_parameters['infer_img']
        img = read_img(img_path)
        
        results = net(fluid.dygraph.to_variable(img))
        lab = np.argsort(results.numpy())
        print("image {} Infer result is: {}".format(img_path, label_dic[lab[0][-1]]))
        

if __name__ == "__main__":
    init_train_parameters()
    infer()