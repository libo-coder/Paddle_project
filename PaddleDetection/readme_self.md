## PaddleDetection


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
数据集根目录需有 VOCdevkit/VOC2007 或 VOCdevkit/VOC2012 文件夹，
该文件夹中需有 `Annotations`,`JPEGImages` 和 `ImageSets/Main` 三个子目录，
`Annotations` 存放图片标注的 `xml` 文件，
`JPEGImages` 存放数据集图片，
`ImageSets/Main` 存放训练 `trainval.txt` 和测试 `test.txt` 列表。