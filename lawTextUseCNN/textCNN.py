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
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)
        self.tokenizer = lambda x: [y for y in x]  # char-level
        self.max_size = 10000
        self.min_freq = 1
        self.UNK, self.PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)  # config.num_classes
        # nn.Sequential()

    def conv_and_pool(self, x, conv):
        x = conv(x)  # bs*1也就是输入图层*seq_len*embedding_dim  example: [128,1,32,300]
        x = F.relu(x)  # bs*输出图层大小    最后一个维度会变成1，这是因为我卷积核的大小是k*embedding_dim    example:[128, 256, 31, 1]
        x = x.squeeze(3)  # 把最后一个维度的1去掉  example:[128, 256, 31]
        x = F.max_pool1d(x, x.size(2))  # example:[128, 256, 1]
        x = x.squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])  # 进来的x[0]就是外面的trains:shape is bs*seq_len   out的输出为bs*seq_len*embedding_dim
        # print("0:", out.size())
        out = out.unsqueeze(1)  # 在1的位置加入一个1，这是因为在textccnn中，我们的输入类别到图像里面是一个灰度图，也就是1个层面的，不是RGB那种三个层面的；
        # print("1:", out.size())
        # 所以上面这个out输出之后的维度是bs*1*seq_len*embedding_dim
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # print("2:", out.size())

        out = self.dropout(out)
        # print("3:", out.size())

        out = self.fc(out)
        # print("4:", out.size())

        return out
