"""
# -*- coding: utf-8 -*-
@author: Hongzhi Fu
"""

import torch
from torch import nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

# 用于编码的CNN模型
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        # 去掉最后一层全连接层
        model = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*model)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    def forward(self, images):
        # 只用于提取特征，不需要计算梯度
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        return features

# 用于解码的RNN模型
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, num_layers, max_seq=20):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq = max_seq
    def forward(self, features, captions, length):
        embedding = self.embed(captions)
        features = features.unsqueeze(1)
        embedding = torch.cat((features, embedding), 1)
        packed = pack_padded_sequence(embedding, length, batch_first=True)
        hidden, _ = self.lstm(packed)
        outputs = self.linear(hidden[0])
        return outputs
    def sample(self, features, states=None):
        # 根据图像生成描述
        sample_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq):
            hidden, states = self.lstm(inputs)
            outputs = self.linear(hidden.squeeze(1))  # 还原成2d
            _, predicted = outputs.max(1)
            sample_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sample_ids = torch.stack(sample_ids, 1)
        return sample_ids




