# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    """"""

    def __init__(self, dataset):
        self.model_name = 'LawClassify'
        self.train_path = dataset + '/OI-dataset/train.txt'
        self.train_label = dataset + '/OI-dataset/train_label.txt'

        self.test_path = dataset + '/OI-dataset/test.txt'
        self.test_label = dataset + '/OI-dataset/test_label.txt'

        self.dev_path = dataset + '/OI-dataset/valid.txt'
        self.dev_label = dataset + '/OI-dataset/valid_label.txt'

        self.class_list = [x.strip() for x in
                           open(dataset + '/OI-dataset/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.pkl'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name

        self.n_vocab = 0

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数

        # self.num_classes = 3  # 类别数
        self.num_epochs = 20  # epoch数
        self.batch_size = 100  # mini-batch大小
        self.pad_size = 50  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed = 300  # 字向量维度

        self.hidden_dim = 300
        self.layer_dim = 1

        self.tokenizer = lambda x: [y for y in x]  # char-level
        self.max_size = 10000
        self.min_freq = 1
        self.UNK, self.PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


class LSTMNet(nn.Module):
    def __init__(self, config):
        super(LSTMNet, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_dim, config.layer_dim, batch_first=True)
        self.fc = nn.Linear(config.hidden_dim, config.num_classes)  # config.num_classes
        # nn.Sequential()

    def forward(self, x):
        out = self.embedding(x[0])  # 进来的x[0]就是外面的trains:shape is bs*seq_len   out的输出为bs*seq_len*embedding_dim
        # print(out.size())
        r_out, (h_n, h_c) = self.lstm(out, None)  # None 表示hidden0使用全0的
        # print(r_out.size())
        # print(h_c.size())
        # print(h_n.size())
        out = self.fc(r_out[:, -1, :])
        # print(out.size())
        return out
