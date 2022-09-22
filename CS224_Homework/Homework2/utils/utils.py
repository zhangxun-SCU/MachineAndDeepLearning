import numpy as np


def normalizeRows(x):
    """
    矩阵归1化：范数归一
    对矩阵的行求范数，在除以范数，（为什么加上一个小的数）
    :param x: 待归一化的矩阵
    :return: 归一化后的矩阵
    """
    n = x.shape[0]
    x /= np.sqrt(np.sum(x**2, axis=1).reshape(n, 1)) + 1e-30
    return x


def softmax(x):
    """
    softmax函数，对每一行进行softmax，压缩为（0， 1）之间的值
    :param x: 矩阵
    :return: 每一行分别经过softmax
    """
    origin_shape = x.shape
    if len(x.shape) > 1:
        # 这是一个矩阵
        tmp = np.max(x, axis=1)  # 求每一行最大值
        x -= tmp.reshape((x.shape[0], 1))  # 每个数据减去此行中的最大值，保证所有值为负值，在softmax后不会出现一个值比其他值大太多
        x = np.exp(x)  # e^x
        tmp = np.sum(x, axis=1)  # 对行求和，得到softmax的分母
        x /= tmp.reshape((x.shape[0], 1))  # 归一化：每个数据除以和
    else:
        # x为向量
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp

    # 断言：softmax不能改变shape
    assert x.shape == origin_shape
    return x
