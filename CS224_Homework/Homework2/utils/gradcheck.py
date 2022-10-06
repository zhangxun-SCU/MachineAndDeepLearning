import numpy as np
import random


def gradcheck_naive(f, x, gradientText):
    """
    检查梯度的函数
    :param f: 一个接收单个输入并输出loss和grad的函数
    :param x: 检查梯度的array
    :param gradientText: 梯度计算的细节
    :return:
    """

    # 随机数本质是伪随机，getstate获取随机状态，传给setstate恢复状态
    randState = random.getstate()
    random.setstate(randState)

    # 计算函数值：loss， grad
    fx, grad = f(x)
    # 用于求梯度的极小值
    h = 1e-4

    # https://blog.csdn.net/weixin_43868107/article/details/102647760
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # 索引元组
        ix = it.multi_index
        x[ix] += h
        random.setstate(randState)
        fxh, _ = f(x)  # f(x+h)
        x[ix] -= 2 * h   # f(x-h)
        random.setstate(randState)
        fxnh, _ = f(x)
        x[ix] += h  # 恢复x的值
        numgrad = (fxh - fxnh) / 2 / h

        # 比较梯度
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))

        if reldiff > 1e-5:
            # 误差过大
            print("Gradient check failed for %s." % gradientText)
            print("First gradient error found at index %s in the vector of gradients" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad))
            return

        # 下一个比较
        it.iternext()
    print("梯度检测通过")
