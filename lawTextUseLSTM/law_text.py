import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time

from RNN_LSTM import *
from utils import *


def sentenceToNumbers(sentence):
    words = []

    for i in sentence:
        words.append(i)
    if len(words) < 50:
        words.extend([PAD] * (50 - len(words)))
    else:
        token = words[:50]

    words_line = []
    for word in words:
        words_line.append(vocab.get(word, vocab.get(UNK)))

    words_line = [words_line]
    words_line = torch.Tensor(words_line).long()
    return (words_line.to(config.device), int(len(words_line)))


if __name__ == '__main__':
    with torch.no_grad():
        dataset = 'data'  # 数据集
        config = Config(dataset)

        lines, labels = readfile(config.train_path, config.train_label)
        vocab = {}

        with open('data/vocab.json', 'r') as f:
            vocab = json.load(f)
        # print(vocab)
        config.n_vocab = len(vocab)
        config.batch_size = 1

        lawClassify = LSTMNet(config)
        lawClassify.eval()
        lawClassify = lawClassify.to(config.device)

        lawClassify.load_state_dict(torch.load('data/saved_dict/trained.pkl'))
        print("Load Success")

        while True:
            sentence = input("输入一句话：")
            if sentence == 'q':
                print("exit")
                break
            data = sentenceToNumbers(sentence)
            # print("data:", data)

            output = lawClassify(data)
            # print(output, output.size())
            chance = F.softmax(output, dim=1)
            print(chance)
            _, d = torch.max(chance, 1)

            index = d.data
            print(index)

            results = []
            with open('data/OI-dataset/class.txt') as f:
                results = f.read().split('\n')

            print(results[index])
