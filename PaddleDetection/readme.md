## PaddleDetection
参考链接： https://paddledetection.readthedocs.io/tutorials/index.html

### 一、fruit 检测训练

#### 1. 数据准备
```bash
python dataset/fruit/download_fruit.py
```

#### 2. 开始训练

训练使用 `yolov3_mobilenet_v1` 基于 COCO 数据集训练好的模型进行 finetune
```bash
python -u tools/train.py -c configs/yolov3_mobilenet_v1_fruit.yml --eval
```

如果想通过 VisualDL 实时观察 loss 和精度值，启动命令添加 `--use_vdl=True`，以及通过 `--vdl_log_dir` 设置日志保存路径，但注意 VisualDL 需 `Python>=3.5`：
```bash
python -u tools/train.py \
       -c configs/yolov3_mobilenet_v1_fruit.yml \
       --use_vdl=True \
       --vdl_log_dir=vdl_fruit_dir/scalar \
       --eval
```

通过 `visualdl` 命令实时查看变化曲线：
```bash
visualdl --logdir vdl_fruit_dir/scalar/ --host <host_IP> --port <port_num>
```

#### 3. 评估预测

**评估命令：**
```bash
python -u tools/eval.py -c configs/yolov3_mobilenet_v1_fruit.yml
```

**预测命令：**
```bash
python -u tools/infer.py \
       -c configs/yolov3_mobilenet_v1_fruit.yml \
       -o weights=https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_fruit.tar \
       --infer_img=demo/orange_71.jpg
```


### 二、训练自定义的数据集

#### 1. 准备数据

**方式一：** 将数据集转换为 COCO 数据集

```bash
python ./ppdet/data/tools/x2coco.py \
       --dataset_type labelme \
       --json_input_dir ./labelme_annos/ \
       --image_input_dir ./labelme_imgs/ \
       --output_dir ./cocome/ \
       --train_proportion 0.8 \
       --val_proportion 0.2 \
       --test_proportion 0.0 \
```

**方式二：** 将数据集转换为 VOC 格式

VOC数据集所必须的文件内容如下所示：
数据集根目录需有 `VOCdevkit/VOC2007` 或 `VOCdevkit/VOC2012` 文件夹，
该文件夹中需有 `Annotations`,`JPEGImages` 和 `ImageSets/Main` 三个子目录，
`Annotations` 存放图片标注的 `xml` 文件，
`JPEGImages` 存放数据集图片，
`ImageSets/Main` 存放训练 `trainval.txt` 和测试 `test.txt` 列表。

```bash
VOCdevkit
├──VOC2007(或VOC2012)
│   ├── Annotations
│       ├── xxx.xml
│   ├── JPEGImages
│       ├── xxx.jpg
│   ├── ImageSets
│       ├── Main
│           ├── trainval.txt
│           ├── test.txt
```

执行命令：
```bash
python dataset/voc/create_list.py -d path/to/dataset
```

#### 2. 选择模型
根据使用场景不同选择合适的模型

#### 3. 修改参数配置
选择好模型后，需要在 configs 目录中找到对应的配置文件，为了适配在自定义数据集上训练，需要对参数配置做一些修改：

- 数据路径配置： 在yaml配置文件中，依据准备好的路径，配置TrainReader、EvalReader和TestReader的路径。

**COCO数据集**
```bash
dataset:
   !COCODataSet
   image_dir: val2017           # 图像数据基于数据集根目录的相对路径
   anno_path: annotations/instances_val2017.json  # 标注文件基于数据集根目录的相对路径
   dataset_dir: dataset/coco    # 数据集根目录
   with_background: true        # 背景是否作为一类标签，默认为true。
```

**VOC数据集**
```bash
dataset:
   !VOCDataSet
   anno_path: trainval.txt      # 训练集列表文件基于数据集根目录的相对路径
   dataset_dir: dataset/voc     # 数据集根目录
   use_default_label: true      # 是否使用默认标签，默认为true。
   with_background: true        # 背景是否作为一类标签，默认为true。
```

**说明：** 如果使用自己的数据集进行训练，需要将 `use_default_label` 设为 `false`，
并在数据集根目录中修改 `label_list.txt` 文件，添加自己的类别名，其中行号对应类别号。

* 类别数修改: 如果自己的数据集类别数和 COCO/VOC 的类别数不同， 需修改yaml配置文件中类别数，num_classes: XX。 
注意：如果 dataset 中设置 `with_background: true`，那么 `num_classes` 数必须是真实类别数+1（背景也算作 1 类）

* 根据需要修改 LearningRate 相关参数:
如果 GPU 卡数变化，依据 lr，batch-size 关系调整 lr: 学习率调整策略
自己数据总数样本数和 COCO 不同，依据 batch_size， 总共的样本数，换算总迭代次数 max_iters，以及 LearningRate 中的 milestones（学习率变化界限）。

* 预训练模型配置：通过在 yaml 配置文件中的 `pretrain_weights: path/to/weights` 参数可以配置路径，可以是链接或权重文件路径。
可直接沿用配置文件中给出的在 ImageNet 数据集上的预训练模型。
同时支持训练在 COCO 或 Obj365 数据集上的模型权重作为预训练模型。


#### 4. 开始训练与部署
* 参数配置完成后，就可以开始训练模型了；
* 训练测试完成后，根据需要可以进行模型部署：
首先需要导出可预测的模型，可参考导出模型教程；导出模型后就可以进行C++预测部署或者python端预测部署。