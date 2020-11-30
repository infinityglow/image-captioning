"""
# -*- coding: utf-8 -*-
@author: Hongzhi Fu
"""

import torch
import os
import pickle
import nltk
import json
import numpy as np
from torch.utils import data
from torchvision import transforms
from build_vocab import Vocabulary

from PIL import Image

class Dataset(data.Dataset):
    def __init__(self, base_dir, part, vocab, transform=None):
        self.img_dir = base_dir + part + "/images/"
        self.cap_dir = base_dir + part + "/captions/"
        self.part = part
        self.ids = os.listdir(self.img_dir)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, idx):
        img_dir = self.img_dir
        cap_dir = self.cap_dir
        vocab = self.vocab

        img_id = str(idx)+".jpg"
        with open(cap_dir+self.part+".json", "r") as f:
            dic = json.load(f)
            captions = dic[img_id]

        image = Image.open(os.path.join(img_dir, img_id)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        image = torch.Tensor(np.array(image))
        target = []
        for caption in captions:
            tokens = nltk.word_tokenize(caption.lower())
            temp = []
            temp.append(vocab("<start>"))
            temp.extend([vocab(token) for token in tokens])
            temp.append(vocab("<end>"))
            tensor = torch.Tensor(temp)
            target.append(tensor)

        return image, target
    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    images, captions = zip(*data)

    images = torch.stack(images, 0)

    lengths = [[len(captions[i][j]) for j in range(len(captions[0]))] for i in range(len(captions))]

    tensor = torch.Tensor(lengths)
    targets = torch.zeros(len(captions), len(captions[0]), int(torch.max(tensor).numpy())).long()

    for i, caption in enumerate(captions):
            for j in range(len(caption)):
                end = lengths[i][j]
                targets[i][j][: end] = caption[j][:]

    return images, targets, lengths

def get_loader(base_dir, part, vocab, transform, batch_size, shuffle, num_workers):

    training_set = Dataset(base_dir, part, vocab, transform)

    data_loader = data.DataLoader(dataset=training_set,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader








