# coding=utf-8
"""
例如 insect 数据集：
    由于该数据集中缺少已标注图片名列表文件 trainval.txt 和 test.txt，所以需要进行生成，
    利用如下 python 脚本，在数据集根目录下执行, 便可生成 trainval.txt 和 test.txt 文件：

1. 将 labelme 标注的所有 xml 文件放入到 VOCdevkit/VOC2007/Annotations 中；
2. 将对应的所有图片放入到 VOCdevkit/VOC2007/JPEGImages 中；
3. 运行 gen_txt.py 生成两个 txt 文件，放入到 VOCdevkit/VOC2007/ImageSets/Main 中。

@author: libo
"""
import os

file_train = open('trainval.txt', 'w')
file_test = open('test.txt', 'w')

for xml_name in os.listdir('train/annotations/xmls'):
    file_train.write(xml_name[:-4] + '\n')

for xml_name in os.listdir('val/annotations/xmls'):
    file_test.write(xml_name[:-4] + '\n')

file_train.close()
file_test.close()
