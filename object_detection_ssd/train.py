# coding=utf-8
"""
模型训练
@author: libo
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
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

from config import train_parameters, init_train_parameters
from reader import multi_process_custom_reader, create_eval_reader
from net import MobileNetSSD


# 初始化日志记录相关参数
def init_log_config():
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, 'train.log')
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)


# 定义优化器。对于训练这种比较大的网络结构，尽量使用阶段性调整学习率的方式
def optimizer_momentum_setting():
    learning_strategy = train_parameters['momentum_strategy']
    learning_rate = fluid.layers.exponential_decay(learning_rate=learning_strategy['learning_rate'],
                                                   decay_steps=learning_strategy['decay_steps'],
                                                   decay_rate=learning_strategy['decay_rate'])
    optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=learning_rate, momentum=0.1)
    return optimizer

#一种自适应的学习率
def optimizer_rms_setting():
    batch_size = train_parameters["train_batch_size"]
    iters = train_parameters["image_count"] // batch_size
    learning_strategy = train_parameters['rsm_strategy']
    lr = learning_strategy['learning_rate']

    boundaries = [i * iters for i in learning_strategy["lr_epochs"]]
    values = [i * lr for i in learning_strategy["lr_decay"]]

    optimizer = fluid.optimizer.RMSProp(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        regularization=fluid.regularizer.L2Decay(0.00005))

    return optimizer


# 配合两种不同数据读取器，定义两种网络构建方法。注意两种定义的时候要共享参数，同时验证网络需要设置为 for_test
def build_train_program_with_async_reader(main_prog, startup_prog):
    with fluid.program_guard(main_prog, startup_prog):
        img = fluid.layers.data(name='img', shape=train_parameters['input_size'], dtype='float32')
        gt_box = fluid.layers.data(name='gt_box', shape=[4], dtype='float32', lod_level=1)
        gt_label = fluid.layers.data(name='gt_label', shape=[1], dtype='int32', lod_level=1)
        difficult = fluid.layers.data(name='difficult', shape=[1], dtype='int32', lod_level=1)

        # 创建一个 Python reader 用于在 python 中提供数据,该函数将返回一个 reader 变量。
        data_reader = fluid.layers.create_py_reader_by_data(capacity=64,    # 缓冲区容量
                                                            feed_list=[img, gt_box, gt_label, difficult],  # 传输数据列表
                                                            name='train')   # reader名称

        # 多进程 reader，使用 python 多进程从 reader 中读取数据
        multi_reader = multi_process_custom_reader(train_parameters['file_list'],
                                                   train_parameters['data_dir'],
                                                   train_parameters['multi_data_reader_count'],
                                                   'train')

        # 将输入数据转换成 reader 返回的多个 mini-batches。每个 mini-batch 分别送入各设备中。
        data_reader.decorate_paddle_reader(multi_reader)
        with fluid.unique_name.guard():
            img, gt_box, gt_label, difficult = fluid.layers.read_file(data_reader)
            model = MobileNetSSD()
            locs, confs, box, box_var = model.net(train_parameters['class_dim'], img, train_parameters['input_size'])
            with fluid.unique_name.guard('train'):
                '''
                locs:预测得到的候选框的位置（中心点的坐标、长、宽）
                confs：每个类别的置信度
                gt_box：groud_truth的位置
                gt_label：ground_tru
                box:候选框的位置
                box_var:方差
                '''
                # paddlepaddle提供了ssd_loss(),返回ssd算法中回归损失和分类损失的加权和
                loss = fluid.layers.ssd_loss(locs, confs, gt_box, gt_label, box, box_var)
                loss = fluid.layers.reduce_sum(loss)
                optimizer = optimizer_rms_setting()
                optimizer.minimize(loss)
                return data_reader, img, loss, locs, confs, box, box_var


def build_eval_program_with_feeder(main_prog, startup_prog, place):
    with fluid.program_guard(main_prog, startup_prog):
        img = fluid.layers.data(name='img', shape=train_parameters['input_size'], dtype='float32')
        gt_box = fluid.layers.data(name='gt_box', shape=[4], dtype='float32', lod_level=1)
        gt_label = fluid.layers.data(name='gt_label', shape=[1], dtype='int32', lod_level=1)
        difficult = fluid.layers.data(name='difficult', shape=[1], dtype='int32', lod_level=1)
        feeder = fluid.DataFeeder(feed_list=[img, gt_box, gt_label, difficult], place=place, program=main_prog)
        reader = create_eval_reader(train_parameters['file_list'], train_parameters['data_dir'], 'eval')
        with fluid.unique_name.guard():
            model = MobileNetSSD()
            locs, confs, box, box_var = model.net(train_parameters['class_dim'], img, train_parameters['input_size'])
            with fluid.unique_name.guard('eval'):
                nmsed_out = fluid.layers.detection_output(locs, confs, box, box_var, nms_threshold=0.45)  # 非极大值抑制得到的结果
                map_eval = fluid.metrics.DetectionMAP(nmsed_out, gt_label, gt_box, difficult,  # 计算map
                                                      train_parameters['class_dim'], overlap_threshold=0.5,
                                                      evaluate_difficult=False, ap_version='11point')
                '''
                “cur_map” 是当前 mini-batch 的 mAP
                "accum_map"是一个 pass 的 mAP 的累加和  
                '''
                cur_map, accum_map = map_eval.get_map_var()
                return feeder, reader, cur_map, accum_map, nmsed_out


# 保存和加载模型。保存时候注意先保存读写参数，可重训练的方式；后保存固化参数，可用于重训练的方式。
# 加载模型有两种，一种是用之前训练的参数，接着全网络继续训练；一种是加载预训练的 mobile-net
def save_model(base_dir, base_name, feed_var_list, target_var_list, train_program, infer_program, exe):
    fluid.io.save_persistables(dirname=base_dir,
                               filename=base_name + '-retrain',
                               main_program=train_program,
                               executor=exe)
    fluid.io.save_inference_model(dirname=base_dir,
                                  params_filename=base_name + '-params',
                                  model_filename=base_name + '-model',
                                  feeded_var_names=feed_var_list,
                                  target_vars=target_var_list,
                                  main_program=infer_program,
                                  executor=exe)


def load_pretrained_params(exe, program):
    retrain_param_file = os.path.join(train_parameters['save_model_dir'], train_parameters['model_prefix'] + '-retrain')
    if os.path.exists(retrain_param_file) and train_parameters['continue_train']:
        logger.info('load param from retrain model')
        print('load param from retrain model')
        fluid.io.load_persistables(executor=exe,
                                   dirname=train_parameters['save_model_dir'],
                                   main_program=program,
                                   filename=train_parameters['model_prefix'] + '-retrain')
    elif train_parameters['pretrained'] and os.path.exists(train_parameters['pretrained_model_dir']):
        logger.info('load param from pretrained model')
        print('load param from pretrained model')

        def if_exist(var):
            return os.path.exists(os.path.join(train_parameters['pretrained_model_dir'], var.name))
        fluid.io.load_vars(exe, train_parameters['pretrained_model_dir'], main_program=program, predicate=if_exist)


def train():
    # 初始化 train_train_parameters 中的参数。class_dim等。
    init_train_parameters()
    print("start ssd, train params:", str(train_parameters))
    logger.info("start ssd, train params: %s", str(train_parameters))

    # 定义设备训练场所
    logger.info("create place, use gpu:" + str(train_parameters['use_gpu']))
    place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()

    # 定义了 program
    logger.info("build network and program")
    train_program = fluid.Program()
    start_program = fluid.Program()
    eval_program = fluid.Program()

    # 构造训练用的 program
    train_reader, img, loss, locs, confs, box, box_var = build_train_program_with_async_reader(train_program, start_program)

    # 构造验证用的program
    eval_feeder, eval_reader, cur_map, accum_map, nmsed_out = build_eval_program_with_feeder(eval_program, start_program, place)
    eval_program = eval_program.clone(for_test=True)

    logger.info("build executor and init params")
    # 创建Executor
    exe = fluid.Executor(place)
    exe.run(start_program)

    # 定义训练、预测的输出值
    train_fetch_list = [loss.name]
    eval_fetch_list = [cur_map.name, accum_map.name]

    # 加载mobilenet预训练的参数到train_program中
    load_pretrained_params(exe, train_program)

    # 获取early_stop参数
    stop_strategy = train_parameters['early_stop']
    successive_limit = stop_strategy['successive_limit']
    sample_freq = stop_strategy['sample_frequency']
    min_curr_map = stop_strategy['min_curr_map']
    min_loss = stop_strategy['min_loss']
    stop_train = False
    total_batch_count = 0
    successive_count = 0
    for pass_id in range(train_parameters["num_epochs"]):
        logger.info("current pass: %d, start read image", pass_id)
        batch_id = 0
        train_reader.start()
        try:
            while True:
                t1 = time.time()
                loss = exe.run(train_program, fetch_list=train_fetch_list)
                period = time.time() - t1
                loss = np.mean(np.array(loss))
                batch_id += 1
                total_batch_count += 1

                if batch_id % 10 == 0:  # 每10个批次打印一次损失
                    logger.info(
                        "Pass {0}, trainbatch {1}, loss {2} time {3}".format(pass_id, batch_id, loss, "%2.2f sec" % period))
                    print(
                        "Pass {0}, trainbatch {1}, loss {2} time {3}".format(pass_id, batch_id, loss, "%2.2f sec" % period))

                if total_batch_count % 400 == 0:  # 每训练400批次的数据，保存一次模型
                    logger.info("temp save {0} batch train result".format(total_batch_count))
                    print("temp save {0} batch train result".format(total_batch_count))
                    fluid.io.save_persistables(dirname=train_parameters['save_model_dir'],  ##从program中取出变量，将其存入指定目录中
                                               filename=train_parameters['model_prefix'] + '-retrain',
                                               main_program=train_program,
                                               executor=exe)

                if total_batch_count == 1 or total_batch_count % sample_freq == 0:  # 满足一定条件，进行一次验证
                    for data in eval_reader():
                        cur_map_v, accum_map_v = exe.run(eval_program, feed=eval_feeder.feed(data), fetch_list=eval_fetch_list)
                        break
                    logger.info("{0} batch train, cur_map:{1} accum_map_v:{2} loss:{3}".format(total_batch_count, cur_map_v[0],
                                                                                               accum_map_v[0], loss))
                    print("{0} batch train, cur_map:{1} accum_map_v:{2} loss:{3}".format(total_batch_count, cur_map_v[0],
                                                                                         accum_map_v[0], loss))
                    # 在验证过程中，map大于所设置的最小的map,或损失小于所设置的最小的损失，认为目标识别正确，successive_count加1
                    if cur_map_v[0] > min_curr_map or loss <= min_loss:
                        successive_count += 1
                        print("successive_count: ", successive_count)
                        fluid.io.save_inference_model(dirname=train_parameters['save_model_dir'],
                                                      params_filename=train_parameters['model_prefix'] + '-params',
                                                      model_filename=train_parameters['model_prefix'] + '-model',
                                                      feeded_var_names=['img'],
                                                      target_vars=[nmsed_out],
                                                      main_program=eval_program,
                                                      executor=exe)
                        # 三次达到验证效果，则停止训练
                        if successive_count >= successive_limit:
                            logger.info("early stop, end training")
                            print("early stop, end training")
                            stop_train = True
                            break
                    else:
                        successive_count = 0
            if stop_train:
                break
        except fluid.core.EOFException:
            train_reader.reset()

    logger.info("training till last epcho, end training")
    print("training till last epcho, end training")
    save_model(train_parameters['save_model_dir'],
               train_parameters['model_prefix'] + '-final',
               ['img'], [nmsed_out], train_program, eval_program, exe)


if __name__ == '__main__':
    # 初始化日志参数。定义全局变量logger,设置了日志文件存放的目录，日志级别等信息。
    init_log_config()
    train()

