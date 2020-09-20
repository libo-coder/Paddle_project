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