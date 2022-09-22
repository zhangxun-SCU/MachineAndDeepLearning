import torch
import numpy as np
from train import train, init_network
from utils import *
from RNN_LSTM import *
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # 数据集
    dataset = f'data'
    config = Config(dataset)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    print("Loading data...")

    # lines, labels = readfile(config.train_path, config.train_label)
    # print(len(lines))
    vocab = build_vocab(config)
    with open('data/vocab.json', 'w') as f:
        json.dump(vocab, f, indent=2)

    # print(vocab)
    train_data = MyDataset(config, config.train_path, config.train_label, vocab)
    dev_data = MyDataset(config, config.dev_path, config.dev_label, vocab)
    test_data = MyDataset(config, config.test_path, config.test_label, vocab)

    train_iter = DataLoader(train_data, batch_size=config.batch_size)
    dev_iter = DataLoader(dev_data, batch_size=config.batch_size)
    test_iter = DataLoader(test_data, batch_size=config.batch_size)
    # train
    config.n_vocab = len(vocab)
    TextCNN_model = LSTMNet(config)
    # 模型放入到GPU中去
    TextCNN_model = TextCNN_model.to(config.device)
    print(TextCNN_model.parameters)
    train(config, TextCNN_model, train_iter, test_iter, dev_iter)
