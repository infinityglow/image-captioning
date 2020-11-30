"""
# -*- coding: utf-8 -*-
@author: Hongzhi Fu
"""

import os
import pickle
import torch
import numpy as np
import torch.nn as nn
import argparse
import json
import matplotlib.pyplot as plt
import random
from torchvision import transforms
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence
from build_vocab import Vocabulary
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN

# hyper-parameters
base_dir = "/home/fhz/Desktop/work/image captioning/flickr8k/data/Flickr8k_Dataset/"
part = "training"
BATCH_SIZE = 5
EPOCH = 5
LR = 0.001
NUM_WORKERS = 2
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1

# torch.cuda.set_device(1) # 用来设置pytorch在哪块GPU上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plotting(loader, args):
    num_imgs = len(loader.dataset)
    img_idx = random.randint(0, num_imgs-1)
    img = loader.dataset[img_idx][0].permute((1, 2, 0))
    with open(args.base_dir+args.part+"/captions/"+args.part+".json", "r") as f:
        dic = json.load(f)
    captions = dic[str(img_idx)+".jpg"]
    plt.imshow(img)
    plt.title(label=captions[random.randint(0, 4)], fontdict={"size": 8})
    plt.show()


def train_main(args):
    if not os.path.exists(args.base_dir+"model/"):
        os.mkdir(args.base_dir+"model/")

    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])


    with open(base_dir + "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    # 新建加载数据集
    loader = get_loader(args.base_dir,
                        args.part, vocab,
                        transform,
                        args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers)
    # 随机显示一张图片和对应标签
    # plotting(loader, args)

    # 实例化编码器和解码器
    encoder = EncoderCNN(args.embed_size)
    decoder = DecoderRNN(args.embed_size, vocab_size, args.hidden_size, args.num_layers, max_seq=20)

    num_captions = 5
    num_examples = len(loader)
    loss_func = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.parameters()) + list(encoder.bn.parameters())
    optimizer = Adam(params, 0.001)

    for epoch in range(args.num_epoch):
        for i, (images, captions, lengths) in enumerate(loader):
            for j in range(num_captions):
                caption = captions[:, j, :]
                length = torch.Tensor(lengths)[:, j]
                length, _ = torch.sort(length, dim=0, descending=True)
                targets = pack_padded_sequence(caption, length, batch_first=True)[0]

                # 正反向传播及优化
                features = encoder(images)
                outputs = decoder(features, caption, length)
                loss = loss_func(outputs, targets)

                decoder.zero_grad()
                encoder.zero_grad()
                loss.backward()

                optimizer.step()
            if i % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}".format(epoch+1, args.num_epoch, i, num_examples, loss.item(), np.exp(loss.item())))
        torch.save(decoder.state_dict(), os.path.join(
            args.model_path, 'decoder-epoch-{}.ckpt'.format(epoch+1)))
        torch.save(encoder.state_dict(), os.path.join(
            args.model_path, 'encoder-epoch-{}.ckpt'.format(epoch+1)))



parser = argparse.ArgumentParser()
# 文件路径参数
parser.add_argument("--base_dir", type=str, default=base_dir)
parser.add_argument("--part", type=str, default=part)

# 超参数
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
parser.add_argument("--num_epoch", type=int, default=EPOCH)
parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
parser.add_argument("--learning_rate", type=int, default=LR)

# 模型参数
parser.add_argument("--embed_size", type=int, default=EMBED_SIZE)
parser.add_argument("--hidden_size", type=int, default=HIDDEN_SIZE)
parser.add_argument("--num_layers", type=int, default=NUM_LAYERS)

config = parser.parse_args(args=[])

train_main(config)

