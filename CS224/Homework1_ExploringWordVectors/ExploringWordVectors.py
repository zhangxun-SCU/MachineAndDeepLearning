import sys

# version_info是一个包含了版本号5个组成部分的元祖，这5个部分分别是主要版本号（major）、次要版本号（minor）、微型版本号（micro）、发布级别（releaselevel）和序列号（serial）
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

# KeyedVectors实现了词向量及其相似性查找, 本质是实体和向量之间的映射
# https://blog.csdn.net/ling620/article/details/99441942
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

# pprint -> pretty print，有两个重要的函数：pprint，pformat
import pprint
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 5]

from nltk.corpus import reuters
import numpy as np
import random
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

import nltk

nltk.download('reuters')

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)


def read_corpus(category="crude"):
    """ 从特定的Reuter目录读取文件
        Params:
            category (string): category name，文件名
        Return:
            list of lists, with words from each of the processed files
    """
    files = reuters.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]
# reuters_corpus = read_corpus()
# pprint.pprint(reuters_corpus[:3], compact=True, width=100)


def distinct_words(corpus):
    """
    得到语料库不同单词的列表
    :param corpus: corpus (list of list of strings): corpus of documents
    :return:corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """

    corpus_words = []
    num_corpus_words = -1

    # ------------------
    # Write your implementation here.
    corpus_words = sorted(list(set([word for sentence in corpus for word in sentence])))
    num_corpus_words = len(corpus_words)
    # ------------------

    return corpus_words, num_corpus_words


def compute_co_occurrence_matrix(corpus, window_size=4):
    """
    由语料库构造一个矩阵M,M矩阵反应词与词之间的练习，考虑中心单词前window_size个单词,后window_size个单词
    :param corpus: list of string, 语料库
    :param window_size: 窗口大小
    :return:
        - 矩阵M：矩阵M表示遍历完语料库中每一句话后，统计的一个次品，它的意义是：行是中心词（按照word2Ind）,
        行内的元素（列）即代表某个词在中心词周围出现的次数，这样M举证就可以反映词与词之间的联系
        - word2Ind：由语料库得到的词-索引表
    """

    words, num_words = distinct_words(corpus)
    M = np.zeros((num_words, num_words))
    # 构造 单词-数字 表
    word2Ind = dict([(word, index) for index, word in enumerate(words)])

    for sentence in corpus:
        current_index = 0
        sentence_len = len(sentence)
        # 句子转换为 数字索引
        indices = [word2Ind[i] for i in sentence]
        while current_index < sentence_len:
            left = max(current_index - window_size, 0)
            right = min(current_index + window_size + 1, sentence_len)
            current_word = sentence[current_index]
            current_word_index = word2Ind[current_word]
            words_around_index = indices[left:current_index] + indices[current_index + 1:right]

            for ind in words_around_index:
                M[current_word_index, ind] += 1

            current_index += 1

    return M, word2Ind
# test_corpus = ["START All that glitters isn't gold END".split(" "), "START All's well that ends well END".split(" ")]
# M_test, word2Ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)
# M_test_ans = np.array(
#     [[0., 0., 0., 1., 0., 0., 0., 0., 1., 0., ],
#      [0., 0., 0., 1., 0., 0., 0., 0., 0., 1., ],
#      [0., 0., 0., 0., 0., 0., 1., 0., 0., 1., ],
#      [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., ],
#      [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., ],
#      [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., ],
#      [0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
#      [0., 0., 0., 0., 0., 1., 1., 0., 0., 0., ],
#      [1., 0., 0., 0., 1., 1., 0., 0., 0., 1., ],
#      [0., 1., 1., 0., 1., 0., 0., 0., 1., 0., ]]
# )
# word2Ind_ans = {'All': 0, "All's": 1, 'END': 2, 'START': 3, 'ends': 4, 'glitters': 5, 'gold': 6, "isn't": 7, 'that': 8,
#                 'well': 9}
# # Test correct word2Ind
# assert (word2Ind_ans == word2Ind_test), "Your word2Ind is incorrect:\nCorrect: {}\nYours: {}".format(word2Ind_ans,
#                                                                                                      word2Ind_test)
# # Test correct M shape
# assert (M_test.shape == M_test_ans.shape), "M matrix has incorrect shape.\nCorrect: {}\nYours: {}".format(M_test.shape,
#                                                                                                           M_test_ans.shape)
#
# # Test correct M values
# for w1 in word2Ind_ans.keys():
#     idx1 = word2Ind_ans[w1]
#     for w2 in word2Ind_ans.keys():
#         idx2 = word2Ind_ans[w2]
#         student = M_test[idx1, idx2]
#         correct = M_test_ans[idx1, idx2]
#         if student != correct:
#             print("Correct M:")
#             print(M_test_ans)
#             print("Your M: ")
#             print(M_test)
#             raise AssertionError(
#                 "Incorrect count at index ({}, {})=({}, {}) in matrix M. Yours has {} but should have {}.".format(idx1,
#                                                                                                                   idx2,
#                                                                                                                   w1,
#                                                                                                                   w2,
#                                                                                                                   student,
#                                                                                                                   correct))
# print(M_test)
# print(word2Ind_test)
# # Print Success
# print("-" * 80)
# print("Passed All Tests!")
# print("-" * 80)


def reduce_to_k_dim(M, k=2):
    """

    :param M:
    :param k:
    :return:
    """
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

        Params:
            M (numpy matrix of shape (number of corpus words, number of corpus words)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """
    n_iters = 10  # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    # ------------------
    # Write your implementation here.
    TSVD = TruncatedSVD(n_components=k, n_iter=n_iters)
    M_reduced = TSVD.fit_transform(M)
    # ------------------

    print("Done.")
    return M_reduced
