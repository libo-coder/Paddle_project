## 项目简介 (未完成)
本项目代码需要使用GPU环境来运行，并且检查相关参数设置, 例如use_gpu, fluid.CUDAPlace(0)等处是否设置正确.

## mobile-net
标准卷积：特点是卷积核的通道数等于输入特征图的通道数

depthwise 卷积：本质就是普通的卷积，只不过采用 1*1 的卷积核，通道数等于特征图的通道数。

采用 depthwise 卷积对不同输入通道分别进行卷积，然后用 pointwise 卷积将上面的输出再进行结合。这样其实整体效果和一个标准卷积是差不多的，但是会大大减少计算量和模型参数量。