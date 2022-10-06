import numpy as np
import random

from utils.gradcheck import gradcheck_naive
from utils.utils import normalizeRows, softmax


def sigmod(x):
    """
    sigmoid
    :param x:
    :return:
    """
    s = 1./(1. + np.exp(-x))
    return s


