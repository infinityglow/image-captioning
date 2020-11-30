"""
# -*- coding: utf-8 -*-
@author: Hongzhi Fu
"""

import json
import argparse
import nltk
import pickle
from collections import Counter

class Vocabulary(object):
    # 构建词和索引之间的对应关系
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.index = 0
    def add(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.index
            self.idx2word[self.index] = word
            self.index += 1
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]
    def __len__(self):
        return len(self.word2idx)

def build_vocab(path, threshold):
    counter = Counter()  # 构建计数器统计单词频率
    # 训练集的构建
    print("开始构建训练集：")
    with open(path, "r") as f1:
        dic = json.load(f1)
        lst = list(dic.values())  # 转换成列表的形式
        for i in range(len(lst)):
            record = lst[i]
            for caption in record:
                token = nltk.word_tokenize(caption.lower())
                counter.update(token)
            if (i+1) % 1000 == 0:
                print("[{}/{}] 单词已标记".format(i+1, len(lst)))
    f1.close()
    print("训练集已标记完成")
    words = [word for word, cnt in counter.items() if cnt>threshold]
    words = words[:-1]
    print(words)
    # 加入到词汇表中
    vocab = Vocabulary()
    vocab.add("<pad>")
    vocab.add("<start>")
    vocab.add("<end>")
    vocab.add("<unk>")

    for word in words:
        vocab.add(word)
    return vocab

def build_vocab_main(args):
    path = args.training_caption_path
    vocab = build_vocab(path, args.threshold)
    vocab_path = args.vocab_path
    print("词汇表的长度为 %d" % len(vocab))
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print("词汇表已保存到 {}".format(vocab_path))


train_caption_path = "/home/fhz/Desktop/work/image captioning/flickr8k/data/Flickr8k_Dataset/training/captions/training.json"
vocab_path = "/home/fhz/Desktop/work/image captioning/flickr8k/data/Flickr8k_Dataset/vocab.pkl"
threshold = 0.4

parser = argparse.ArgumentParser()
parser.add_argument("--training_caption_path", type=str, default=train_caption_path)
parser.add_argument("--vocab_path", type=str, default=vocab_path)
parser.add_argument("--threshold", type=str, default=2)
config = parser.parse_args(args=[])

if __name__ == '__main__':
    build_vocab_main(config)
