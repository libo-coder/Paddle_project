# coding=utf-8
"""
模型训练
@author: libo
"""
import paddle.fluid as fluid
from net import DenseNet
import numpy as np
import paddle
import reader
import os
import logging
from config import train_parameters, init_train_parameters
import time


def init_log_config():
    """ 初始化日志相关配置 """
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, 'train.log')
    sh = logging.StreamHandler()
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)


def optimizer_momentum_setting():
    """  
    阶梯型的学习率适合比较大规模的训练数据  
    """
    learning_strategy = train_parameters['momentum_strategy']
    batch_size = train_parameters["train_batch_size"]
    iters = train_parameters["image_count"] // batch_size
    lr = learning_strategy['learning_rate']

    boundaries = [i * iters for i in learning_strategy["lr_epochs"]]
    values = [i * lr for i in learning_strategy["lr_decay"]]
    learning_rate = fluid.layers.piecewise_decay(boundaries, values)
    optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    return optimizer


def optimizer_rms_setting(parameters):
    """  
    阶梯型的学习率适合比较大规模的训练数据  
    """
    batch_size = train_parameters["train_batch_size"]
    iters = train_parameters["image_count"] // batch_size
    learning_strategy = train_parameters['rsm_strategy']
    lr = learning_strategy['learning_rate']

    boundaries = [i * iters for i in learning_strategy["lr_epochs"]]
    values = [i * lr for i in learning_strategy["lr_decay"]]

    optimizer = fluid.optimizer.RMSProp(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        parameter_list=parameters)

    return optimizer


def optimizer_sgd_setting():
    """  
    loss下降相对较慢，但是最终效果不错，阶梯型的学习率适合比较大规模的训练数据  
    """
    learning_strategy = train_parameters['sgd_strategy']
    batch_size = train_parameters["train_batch_size"]
    iters = train_parameters["image_count"] // batch_size
    lr = learning_strategy['learning_rate']

    boundaries = [i * iters for i in learning_strategy["lr_epochs"]]
    values = [i * lr for i in learning_strategy["lr_decay"]]
    learning_rate = fluid.layers.piecewise_decay(boundaries, values)
    optimizer = fluid.optimizer.SGD(learning_rate=learning_rate)
    return optimizer


def optimizer_adam_setting():
    """  
    能够比较快速的降低 loss，但是相对后期乏力  
    """
    learning_strategy = train_parameters['adam_strategy']
    learning_rate = learning_strategy['learning_rate']
    optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)
    return optimizer


def eval_net(reader, model):
    acc_set = []

    for batch_id, data in enumerate(reader()):
        dy_x_data = np.array([x[0] for x in data]).astype('float32')
        y_data = np.array([x[1] for x in data]).astype('int')
        y_data = y_data[:, np.newaxis]
        img = fluid.dygraph.to_variable(dy_x_data)
        label = fluid.dygraph.to_variable(y_data)
        label.stop_gradient = True
        prediction, acc = model(img, label)

        acc_set.append(float(acc.numpy()))

        # get test acc and loss
    acc_val_mean = np.array(acc_set).mean()

    return acc_val_mean


def train():
    with fluid.dygraph.guard():
        epoch_num = train_parameters["num_epochs"]
        net = DenseNet("densenet", layers=121, dropout_prob=train_parameters['dropout_prob'],
                       class_dim=train_parameters['class_dim'])
        optimizer = optimizer_rms_setting(net.parameters())
        file_list = os.path.join(train_parameters['data_dir'], "train.txt")
        train_reader = paddle.batch(reader.custom_image_reader(file_list, train_parameters['data_dir'], 'train'),
                                    batch_size=train_parameters['train_batch_size'],
                                    drop_last=True)
        test_reader = paddle.batch(reader.custom_image_reader(file_list, train_parameters['data_dir'], 'val'),
                                   batch_size=train_parameters['train_batch_size'],
                                   drop_last=True)
        if train_parameters["continue_train"]:
            model, _ = fluid.dygraph.load_dygraph(train_parameters["save_persistable_dir"])
            net.load_dict(model)

        best_acc = 0
        for epoch_num in range(epoch_num):

            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                y_data = np.array([x[1] for x in data]).astype('int')
                y_data = y_data[:, np.newaxis]

                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True
                t1 = time.time()
                out, acc = net(img, label)
                t2 = time.time()
                forward_time = t2 - t1
                loss = fluid.layers.cross_entropy(out, label)
                avg_loss = fluid.layers.mean(loss)
                # dy_out = avg_loss.numpy()
                t3 = time.time()
                avg_loss.backward()
                t4 = time.time()
                backward_time = t4 - t3
                optimizer.minimize(avg_loss)
                net.clear_gradients()
                # print(forward_time, backward_time)

                dy_param_value = {}
                for param in net.parameters():
                    dy_param_value[param.name] = param.numpy

                if batch_id % 40 == 0:
                    logger.info("Loss at epoch {} step {}: {}, acc: {}".format(epoch_num, batch_id, avg_loss.numpy(), acc.numpy()))

            net.eval()
            epoch_acc = eval_net(test_reader, net)
            net.train()
            if epoch_acc > best_acc:
                fluid.dygraph.save_dygraph(net.state_dict(), train_parameters["save_persistable_dir"])
                fluid.dygraph.save_dygraph(optimizer.state_dict(), train_parameters["save_persistable_dir"])
                best_acc = epoch_acc
                logger.info("model saved at epoch {}, best accuracy is {}".format(epoch_num, best_acc))
        logger.info("Final loss: {}".format(avg_loss.numpy()))


if __name__ == "__main__":
    init_log_config()
    init_train_parameters()
    train()
