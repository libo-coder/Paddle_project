import uuid
from PIL import Image, ImageEnhance, ImageDraw
from config import train_parameters, init_train_parameters


# 为了更直观的看到训练样本的形态，增加打印图片，并画出 bbox 的函数
def log_feed_image(img, sampled_labels):
    draw = ImageDraw.Draw(img)
    target_h = train_parameters['input_size'][1]
    target_w = train_parameters['input_size'][2]
    for label in sampled_labels:
        print(label)
        draw.rectangle((label[1] * target_w, label[2] * target_h, label[3] * target_w, label[4] * target_h), None, 'red')
    img.save(str(uuid.uuid1()) + '.jpg')


########################### 定义训练时候，数据增强需要的辅助类，例如外接矩形框、采样器 ###################
class sampler:
    def __init__(self, max_sample, max_trial, min_scale, max_scale,
                 min_aspect_ratio, max_aspect_ratio, min_jaccard_overlap,
                 max_jaccard_overlap):
        self.max_sample = max_sample
        self.max_trial = max_trial
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_jaccard_overlap = min_jaccard_overlap
        self.max_jaccard_overlap = max_jaccard_overlap


class bbox:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax