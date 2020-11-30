"""
# -*- coding: utf-8 -*-
@author: Hongzhi Fu
"""

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from torchvision import transforms
from PIL import Image
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN


# 设备配置
# torch.cuda.set_device(1) # 这句用来设置pytorch在哪块GPU上运行
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_img(img_path, transform=None):
    image = Image.open(img_path)
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image

def test(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    encoder = EncoderCNN(args.embed_size).eval()
    decoder = DecoderRNN(args.embed_size, len(vocab), args.hidden_size, args.num_layers)

    # 加载训练好的模型的参数
    encoder.load_state_dict(torch.load(args.encoder_path, map_location='cpu'))
    decoder.load_state_dict(torch.load(args.decoder_path, map_location='cpu'))

    image = load_img(args.img_path, transform)

    feature = encoder(image)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()

    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    print(sentence)
    image = Image.open(args.img_path)
    plt.imshow(np.asarray(image))
    plt.show()

parse = argparse.ArgumentParser()
parse.add_argument("--img_path", type=str,
                   default="/home/fhz/Desktop/work/image captioning/flickr8k/data/Flickr8k_Dataset/training/images/0.jpg")
parse.add_argument("--encoder_path", type=str,
                   default="/home/fhz/Desktop/work/image captioning/flickr8k/data/Flickr8k_Dataset/model/encoder-epoch-10.ckpt")
parse.add_argument("--decoder_path", type=str,
                   default="/home/fhz/Desktop/work/image captioning/flickr8k/data/Flickr8k_Dataset/model/decoder-epoch-10.ckpt")
parse.add_argument("--vocab_path", type=str,
                   default="/home/fhz/Desktop/work/image captioning/flickr8k/data/Flickr8k_Dataset/vocab.pkl")

parse.add_argument("--embed_size", type=int, default=256)
parse.add_argument("--hidden_size", type=int, default=512)
parse.add_argument("--num_layers", type=int, default=1)

config = parse.parse_args(args=[])
test(config)

