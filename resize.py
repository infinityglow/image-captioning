"""
# -*- coding: utf-8 -*-
@author: Hongzhi Fu
"""

import os
import argparse
from PIL import Image

def resize(img_dir, out_dir, size):
    # 不存在就建立新文件夹
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    images = os.listdir(img_dir)
    num_images = len(images)
    print("开始图片尺寸修改：")
    for i, image in enumerate(images):
        with open(os.path.join(img_dir, image), "r+b") as f:
            with Image.open(f) as img:
                img = img.resize(size, Image.ANTIALIAS)
                img.save(os.path.join(out_dir, image), img.format)
        if (i+1) % 100 == 0:
            print("已完成 [{}/{}]，保存在{}".format(i+1, num_images, out_dir))

def resize_main(args):
    img_size = [args.size, args.size]
    resize(args.img_dir, args.out_dir, img_size)
    print("图片尺寸修改完成")

image_dir = "/home/fhz/Desktop/work/image captioning/flickr8k/data/Flickr8k_Dataset/Flickr8k_Dataset/"
out_dir = "/home/fhz/Desktop/work/image captioning/flickr8k/data/Flickr8k_Dataset/resized/"
size = 256

parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str, default=image_dir)
parser.add_argument("--out_dir", type=str, default=out_dir)
parser.add_argument("--size", type=int, default=size)
config = parser.parse_args(args=[])

if __name__ == "__main__":
    resize_main(config)