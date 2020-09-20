# coding=utf-8
"""
模型评估
@author: libo
"""
import paddle.fluid as fluid
import paddle
import reader
from net import DenseNet
import numpy as np
import os
from config import train_parameters, init_train_parameters


def eval():
    
    file_list = os.path.join(train_parameters['data_dir'], "eval.txt")
    with fluid.dygraph.guard():
        model, _ = fluid.dygraph.load_dygraph(train_parameters["save_persistable_dir"])
        net = DenseNet("densenet", layers = 121, dropout_prob = train_parameters['dropout_prob'], class_dim = train_parameters['class_dim'])
        net.load_dict(model)
        net.eval()
        test_reader = paddle.batch(reader.custom_image_reader(file_list, reader.train_parameters['data_dir'], 'val'),
                                batch_size=train_parameters['train_batch_size'],
                                drop_last=True)
        accs = []
        for batch_id, data in enumerate(test_reader()):
            dy_x_data = np.array([x[0] for x in data]).astype('float32')
            y_data = np.array([x[1] for x in data]).astype('int')
            y_data = y_data[:, np.newaxis]
            
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)
            label.stop_gradient = True
    
            out, acc = net(img, label)
            lab = np.argsort(out.numpy())
            #print(batch_id, label.numpy()[0][0], lab[0][-1])
            accs.append(acc.numpy()[0])
    print(np.mean(accs))
    
if __name__ == "__main__":
    init_train_parameters()
    eval()