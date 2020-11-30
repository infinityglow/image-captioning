"""
# -*- coding: utf-8 -*-
@author: Hongzhi Fu
"""

import os
import shutil
import argparse
import json

def read(args):
    # 分别读取三个txt文档（包含图片名称）
    with open(args.training, "r") as f1:
        training_set_img = f1.read().splitlines()
    with open(args.validation, "r") as f2:
        validation_set_img = f2.read().splitlines()
    with open(args.test, "r") as f3:
        test_set_img = f3.read().splitlines()
    return [training_set_img, validation_set_img, test_set_img]

def split_dataset_img(path, base_path, img_path):
    train, validation, test = path[0], path[1], path[2]
    # 如果不存在就创建一个目录
    if not os.path.exists(base_path + "/training"):
        os.mkdir(base_path + "/training")
    if not os.path.exists(base_path + "/training/images"):
        os.mkdir(base_path + "/training/images")
    for i in range(len(train)):
            src = os.path.join(img_path, train[i])
            dst = os.path.join(base_path+"training/images/"+str(i)+".jpg")
            shutil.copy(src, dst)
    for j in range(len(validation)):
            src = os.path.join(img_path, validation[j])
            dst = os.path.join(base_path+"training/images/"+str(len(train)+j)+".jpg")
            shutil.copy(src, dst)
    print("已完成 {} 图像数据集的分割，共有 {} 张图片".format("training", len(train)+len(validation)))

    if not os.path.exists(base_path + "/test"):
        os.mkdir(base_path + "/test")
    if not os.path.exists(base_path + "/test/images"):
        os.mkdir(base_path + "/test/images")
    for k in range(len(test)):
            src = os.path.join(img_path, test[k])
            dst = os.path.join(base_path+"test/images/"+str(k)+".jpg")
            shutil.copy(src, dst)
    print("已完成 {} 图像数据集的分割，共有 {} 张图片".format("test", len(test)))

def read_cap(cap_path):
    with open(cap_path+"Flickr8k.token.txt", "r") as f:
        captions = f.read().splitlines()
    return captions

def process(captions):
    # 创建一个字典，组成对应关系
    dic = dict()
    cache = ""
    temp_list = []
    for i in range(len(captions)):
        segment = captions[i].split(maxsplit=1)
        img = segment[0][:-2]; cap = segment[-1]
        if cache != img:
            if temp_list:
                dic[cache] = temp_list
            cache = img  # 暂存图片的名称
            temp_list = []  # 如果不是同一张图片就清空列表
        temp_list.append(cap)
    dic[cache] = temp_list
    return dic

def to_json(pack, base_path):
    mapping = {0: "training", 1: "test"}
    for i in range(len(pack)):
        if not os.path.exists(base_path + mapping[i]):
            os.mkdir(base_path + mapping[i])
        if not os.path.exists(base_path + mapping[i] + "/captions"):
            os.mkdir(base_path + mapping[i] + "/captions")
        with open(base_path+mapping[i]+"/captions/"+mapping[i]+".json", "w") as f:
            json.dump(pack[i], f, ensure_ascii=False, indent=4, separators=(',', ':\n\t'))
            f.write("\n")
        f.close()
        print("已完成 {} 标注数据集的分割，共有 {} 条记录".format(mapping[i], len(pack[i])))

def split_dataset_cap(dic, names, base_path):
    training = dict(); test = dict()
    names_train = names[0]; names_validation = names[1]; names_test = names[2]
    # 训练集
    for i in range(len(names_train)):
        if names_train[i] in dic.keys():
            training[str(i)+".jpg"] = dic.get(names_train[i])
    for i in range(len(names_validation)):
        if names_validation[i] in dic.keys():
            training[str(len(names_train)+i)+".jpg"] = dic.get(names_validation[i])
    # 测试集
    for i in range(len(names_test)):
        if names_test[i] in dic.keys():
            test[str(i)+".jpg"] = dic.get(names_test[i])
    to_json([training, test], base_path)  # 将字典转化为json的形式，并写入

def split_main(args):
    # 分割图像数据集
    print("开始进行图像数据集的分割：\n")
    three_sets_img = read(args)
    split_dataset_img(three_sets_img, args.base_path, args.img_path)
    # 分割标注数据集
    print("\n开始进行标注数据集的分割：\n")
    captions = read_cap(args.cap_path)
    dic = process(captions)
    split_dataset_cap(dic, three_sets_img, args.base_path)


base_path = "/home/fhz/Desktop/work/image captioning/flickr8k/data/Flickr8k_Dataset/"
img_path = "/home/fhz/Desktop/work/image captioning/flickr8k/data/Flickr8k_Dataset/resized/"
cap_path = "/home/fhz/Desktop/work/image captioning/flickr8k/data/Flickr8k_text/"
training = cap_path + "Flickr_8k.trainImages.txt"
validation = cap_path + "Flickr_8k.devImages.txt"
test = cap_path + "Flickr_8k.testImages.txt"

parser = argparse.ArgumentParser()
parser.add_argument("--training", type=str, default=training)
parser.add_argument("--validation", type=str, default=validation)
parser.add_argument("--test", type=str, default=test)
parser.add_argument("--base_path", type=str, default=base_path)
parser.add_argument("--img_path", type=str, default=img_path)
parser.add_argument("--cap_path", type=str, default=cap_path)

config = parser.parse_args(args=[])

if __name__ == '__main__':
    split_main(config)
