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