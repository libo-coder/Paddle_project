# coding=utf-8
"""
DenseNet网络模型结构
@author: libo
"""
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear

class BNConvLayer(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_filters, num_channels, filter_size, stride=1, groups=1, act='relu'):
        super(BNConvLayer, self).__init__(name_scope)

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False)

        self._batch_norm = BatchNorm(num_channels, act=act)

    def forward(self, inputs):
        y = self._batch_norm(inputs)
        y = self._conv(y)
        return y


class BottleneckLayer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 num_channels,
                 drop_out_prob):
        super(BottleneckLayer, self).__init__(name_scope)

        self.bn_conv1 = BNConvLayer(
            self.full_name(),
            num_filters=num_filters * 4,
            num_channels=num_channels,
            filter_size=1)

        self.bn_conv2 = BNConvLayer(
            self.full_name(),
            num_filters=num_filters,
            num_channels=num_filters * 4,
            filter_size=3)

        self.drop_out_prob = drop_out_prob

    def forward(self, inputs):
        y = self.bn_conv1(inputs)
        y = fluid.layers.dropout(x=y, dropout_prob=self.drop_out_prob)
        y = self.bn_conv2(y)
        y = fluid.layers.dropout(x=y, dropout_prob=self.drop_out_prob)
        return y


class DenseBlock(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 num_channels,
                 block_num,
                 drop_out_prob):
        super(DenseBlock, self).__init__(name_scope)

        self.block = BottleneckLayer(
            self.full_name(),
            num_filters=num_filters,
            num_channels=num_channels,
            drop_out_prob=drop_out_prob)

        self.block_num = block_num
        channels = num_channels + num_filters
        self.convs = []
        for i in range(self.block_num - 1):
            conv_block = self.add_sublayer(
                'bb_%d_%d' % (i, i),
                BottleneckLayer(
                    self.full_name(),
                    num_filters=num_filters,
                    num_channels=channels,
                    drop_out_prob=drop_out_prob))
            self.convs.append(conv_block)
            channels = channels + num_filters

        self.out_channel = channels

    def forward(self, inputs):
        layers = []
        layers.append(inputs)
        y = self.block(inputs)
        layers.append(y)
        for conv in self.convs:
            y = paddle.fluid.layers.concat(layers, axis=1)
            y = conv(y)
            layers.append(y)
        y = paddle.fluid.layers.concat(layers, axis=1)
        return y


class TransitionLayer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 num_channels,
                 drop_out_prob):
        super(TransitionLayer, self).__init__(name_scope)

        self.conv = BNConvLayer(
            self.full_name(),
            num_filters=num_filters,
            num_channels=num_channels,
            filter_size=1)

        self.pool2d = Pool2D(
            pool_size=2,
            pool_stride=2,
            pool_type='avg')

        self.dropout_prob = drop_out_prob

    def forward(self, inputs):
        y = self.conv(inputs)
        y = fluid.layers.dropout(x=y, dropout_prob=self.dropout_prob)
        y = self.pool2d(y)
        return y


class LoopLayer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 num_channels,
                 block_num,
                 drop_out_prob):
        super(LoopLayer, self).__init__(name_scope)

        self.denseblock = DenseBlock(
            self.full_name(),
            num_filters=num_filters,
            num_channels=num_channels,
            block_num=block_num,
            drop_out_prob=drop_out_prob)

        self.channel = self.denseblock.out_channel

        self.transblock = TransitionLayer(
            self.full_name(),
            num_filters=num_filters,
            num_channels=self.channel,
            drop_out_prob=drop_out_prob)

    def forward(self, inputs):
        y = self.denseblock(inputs)
        y = self.transblock(y)
        return y


class DenseNet(fluid.dygraph.Layer):
    def __init__(self, name_scope, layers, dropout_prob, class_dim=5):
        super(DenseNet, self).__init__(name_scope)

        self.layers = layers
        self.dropout_prob = dropout_prob

        layer_count_dict = {
            121: (32, [6, 12, 24, 16]),
            169: (32, [6, 12, 32, 32]),
            201: (32, [6, 12, 48, 32]),
            161: (48, [6, 12, 36, 24])
        }
        layer_conf = layer_count_dict[self.layers]

        self.conv1 = Conv2D(
            num_channels=3,
            num_filters=layer_conf[0] * 2,
            filter_size=7,
            stride=2,
            padding=3,
            groups=1,
            act=None,
            bias_attr=False)

        self.pool1 = Pool2D(
            pool_size=3,
            pool_padding=1,
            pool_stride=2,
            pool_type='max')
        channels = layer_conf[0] * 2

        self.convs = []
        for i in range(len(layer_conf[1]) - 1):
            conv_block = self.add_sublayer(
                'bb_%d_%d' % (i, i),
                LoopLayer(
                    self.full_name(),
                    num_filters=layer_conf[0],
                    num_channels=channels,
                    block_num=layer_conf[1][i],
                    drop_out_prob=self.dropout_prob))
            channels = layer_conf[0]
            self.convs.append(conv_block)

        self.conv3 = DenseBlock(
            self.full_name(),
            num_filters=layer_conf[1][-1],
            num_channels=layer_conf[0],
            block_num=layer_conf[0],
            drop_out_prob=self.dropout_prob)

        self.pool2 = Pool2D(
            global_pooling=True,
            pool_type='avg')

        self.fc = Linear(input_dim=544,
                         output_dim=class_dim,
                         act='softmax')

    def forward(self, inputs, label=None):
        y = self.conv1(inputs)
        y = self.pool1(y)
        #   print(len(self.convs))
        for conv in self.convs:
            y = conv(y)
        y = self.conv3(y)
        y = self.pool2(y)
        y = fluid.layers.reshape(y, [-1, 544])
        y = self.fc(y)
        if label is not None:
            acc = fluid.layers.accuracy(input=y, label=label)
            return y, acc
        else:
            return y
