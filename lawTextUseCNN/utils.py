# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from torch.utils.data import Dataset
import json

UNK, PAD = '<UNK>', '<PAD>'


def readfile(datapath, labelpath):
    source = open(datapath, 'r', encoding='UTF-8-sig').read()
    lines = source.split('\n')
    docs = [line for line in lines if len(line) > 0]

    source = open(labelpath, 'r', encoding='UTF-8-sig').read()
    source = source.replace('\xef\xbb\xbf', '')
    lines_label = source.split('\n')[:-1]
    labels = [[int(x) for x in label.split()] for label in lines_label if len(label) > 0]
    labels = np.array(labels, dtype='int')
    labels = labels[:, 8]
    labels = labels + 1
    assert len(docs) == len(labels)
    # assert labels.shape[1] == 15

    return docs, labels


def build_vocab(config, maskid=0):
    file_path = config.train_path
    file_path = config.train_path
    tokenizer = config.tokenizer
    max_size = config.max_size
    min_freq = config.min_freq
    vocab_dic = {}

    # lines = [' '.join(line) for line in lines]
    # wordset = set(item for line in lines for item in line.strip().split())
    # vocab_dic = {word: index + 1 for index, word in enumerate(wordset)}
    # vocab_dic['<mask>'] = maskid
    # vocab_dic[PAD] = len(vocab_dic)
    # vocab_dic[UNK] = len(vocab_dic) + 1

    with open(file_path, 'r', encoding='UTF-8') as f:  # 读取数据文件：训练集还是验证集还是测试集
        for line in tqdm(f):  # 读取每行的数据
            lin = line.strip()  # 去掉最后的\n符号
            if not lin:  # 如果是空的话，直接continue跳过
                continue
            content = lin.split('\t')[0]  # 文本数据用\t进行分割，取第一个[0]是文本，第二个是【1】是标签数据

            for word in tokenizer(content):  # 用字进行切割，tokenizer(content)看函数得到的是一个列表对吧。

                vocab_dic[word] = vocab_dic.get(word, 0) + 1  # 生成词表字典，这个字典的get就是有这个元素就返回结果，数量在这里原始值加1，如果没有返回默认值为0，数量加1；
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]  # 首先是根据频次筛选，然后sort一下降序，然后取词表最大

        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}  # 从词表字典中找到我们需要的那些就可以了
        vocab_dic.update({config.UNK: len(vocab_dic), config.PAD: len(vocab_dic) + 1, 'maskid': maskid})  # 然后更新两个字符，一个是unk字符，一个pad字符

    return vocab_dic


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


class MyDataset(Dataset):
    def __init__(self, config, datapath, labelpath, vocab_file):

        self.docs, self.labels = readfile(datapath, labelpath)
        self.config = config
        self.pad_size = config.pad_size
        self.tokenizer = config.tokenizer
        self.vocab = vocab_file
        self.device = config.device

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):

        docs = [' '.join(line) for line in self.docs]
        content, label = docs[idx], self.labels[idx]

        token = content.strip().split()
        seq_len = len(token)
        words_line = []

        if len(token) < self.pad_size:
            token.extend([PAD] * (self.pad_size - len(token)))
        else:
            token = token[:self.pad_size]
        seq_len = self.pad_size

        for word in token:
            words_line.append(self.vocab.get(word, self.vocab.get(UNK)))

        data = torch.Tensor(words_line).long()
        label = torch.tensor(label).long()
        seq_len = int(seq_len)
        return (data.to(self.device), seq_len), label.to(self.device)
