# 中文OCR模型快速使用

## 1.环境配置

请先参考[快速安装](./installation.md)配置PaddleOCR运行环境。

*注意：也可以通过 whl 包安装使用PaddleOCR，具体参考[Paddleocr Package使用说明](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/whl.md)。*

## 2.inference模型下载

### 超轻量级中文OCR模型
**检测模型：**
* [inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db_infer.tar)
* [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db.tar)

**方向分类器模型：**
* [inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_infer.tar)
* [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn.tar)

**识别模型：**
* [inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance_infer.tar)
* [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance.tar)

### 通用中文OCR模型
**检测模型：**
* [inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db_infer.tar)
* [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db.tar)

**方向分类器模型：**
* [inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_infer.tar)
* [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn.tar)

**识别模型：**
* [inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance_infer.tar)
* [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance.tar)


解压完毕后应有如下文件结构：
```
|-inference
    |-ch_rec_mv3_crnn
        |- model
        |- params
    |-ch_det_mv3_db
        |- model
        |- params
    ...
```

## 3.单张图像或者图像集合预测

以下代码实现了文本检测、识别串联推理，在执行预测时，需要通过参数image_dir指定单张图像或者图像集合的路径、
参数det_model_dir指定检测inference模型的路径
参数rec_model_dir指定识别inference模型的路径。
可视化识别结果默认保存到 ./inference_results 文件夹里面。

```bash

# 预测image_dir指定的单张图像
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/ch_det_mv3_db/"  --rec_model_dir="./inference/ch_rec_mv3_crnn/"

# 预测image_dir指定的图像集合
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/" --det_model_dir="./inference/ch_det_mv3_db/"  --rec_model_dir="./inference/ch_rec_mv3_crnn/"

# 如果想使用CPU进行预测，需设置use_gpu参数为False
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/ch_det_mv3_db/"  --rec_model_dir="./inference/ch_rec_mv3_crnn/" --use_gpu=False
```

- 通用中文OCR模型

请按照上述步骤下载相应的模型，并且更新相关的参数，示例如下：
```
# 预测image_dir指定的单张图像
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/ch_det_r50_vd_db/"  --rec_model_dir="./inference/ch_rec_r34_vd_crnn/"
```

- 支持空格的通用中文OCR模型

请按照上述步骤下载相应的模型，并且更新相关的参数，示例如下：

*注意：请将代码更新到最新版本，并添加参数 `--use_space_char=True` *

```
# 预测image_dir指定的单张图像
python3 tools/infer/predict_system.py --image_dir="./doc/imgs_en/img_12.jpg" --det_model_dir="./inference/ch_det_r50_vd_db/"  --rec_model_dir="./inference/ch_rec_r34_vd_crnn_enhance/" --use_space_char=True
```

更多的文本检测、识别串联推理使用方式请参考文档教程中[基于Python预测引擎推理](./inference.md)。

此外，文档教程中也提供了中文OCR模型的其他预测部署方式：
- [基于C++预测引擎推理](../../deploy/cpp_infer/readme.md)
- [服务部署](./serving.md)
- [端侧部署](../../deploy/lite/readme.md)