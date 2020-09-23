## 模块介绍
PaddleDetection 的数据处理模块，所有的代码逻辑都在 `ppdet/data` 中，数据处理模型用于加载数据并将其转换成适用于物体检测模型的训练、评估、推理所需要的格式。

**数据处理模模块的主要构成如下架构所示：**
```bash
ppdet/data/
  ├── reader.py     # 数据处理模块的总接口
  ├── shared_queue  # 共享内存管理模块
  │   ├── queue.py        # 定义共享内存队列
  │   ├── sharedmemory.py # 负责分配内存
  ├── source  # 数据源管理模块
  │   ├── dataset.py      # 定义数据源基类，各类数据集继承于此
  │   ├── coco.py         # COCO数据集解析与格式化数据
  │   ├── voc.py          # Pascal VOC数据集解析与格式化数据
  │   ├── widerface.py    # WIDER-FACE数据集解析与格式化数据
  ├── tests  # 单元测试模块
  │   ├── test_dataset.py # 对数据集解析、加载等进行单元测试
  │   │   ...
  ├── tools  # 一些有用的工具
  │   ├── x2coco.py       # 将其他数据集转换为COCO数据集格式
  ├── transform  # 数据预处理模块
  │   ├── batch_operators.py  # 定义各类基于批量数据的预处理算子
  │   ├── op_helper.py    # 预处理算子的辅助函数
  │   ├── operators.py    # 定义各类基于单张图片的预处理算子
  ├── parallel_map.py     # 在多进程/多线程模式中对数据预处理操作进行加速
```
