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
    """
    从Reuter里读取文件
    :param category:
    :return: list of lists，每个处理过的文件的词语构成的列表
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

    corpus_words = sorted(list(set([word for sentence in corpus for word in sentence])))
    num_corpus_words = len(corpus_words)

    return corpus_words, num_corpus_words


def compute_co_occurrence_matrix(corpus, window_size=4):
    """
    由语料库构造一个矩阵M,M矩阵反应词与词之间的练习，考虑中心单词前window_size个单词,后window_size个单词
    :param corpus: list of string, 语料库
    :param window_size: 窗口大小
    :return:
        - 矩阵M：矩阵M表示遍历完语料库中每一句话后，统计的一个次品，它的意义是：行是中心词（按照word2Ind）,
        行内的元素（列）即代表某个词在中心词周围出现的次数，这样M举证就可以反映词与词之间的联系，换言之，
        行就可以表示每个单词了，因为每个数都反映了这个词与其他的共同出现的次数
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
    构造一个在矩阵上执行降维以生成 k 维嵌入的方法。使用 SVD 获取前 k 个分量并生成新的 k 维嵌入矩阵
    :param M:用于降维的矩阵
    :param k:降维后每个词的维度
    :return:M_reduced：降维后的矩阵，形状（单词个数/num_word, k）
    """
    n_iters = 10  # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    # 参考资料
    # https://scikit-learn.org.cn/view/612.html
    # https://zhuanlan.zhihu.com/p/134512367
    TSVD = TruncatedSVD(n_components=k, n_iter=n_iters)
    M_reduced = TSVD.fit_transform(M)

    print("Done.")
    return M_reduced
# # Define toy corpus and run student code
# test_corpus = ["START All that glitters isn't gold END".split(" "), "START All's well that ends well END".split(" ")]
# M_test, word2Ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)
# M_test_reduced = reduce_to_k_dim(M_test, k=2)
#
# # Test proper dimensions
# assert (M_test_reduced.shape[0] == 10), "M_reduced has {} rows; should have {}".format(M_test_reduced.shape[0], 10)
# assert (M_test_reduced.shape[1] == 2), "M_reduced has {} columns; should have {}".format(M_test_reduced.shape[1], 2)
# print(M_test_reduced)
# # Print Success
# print ("-" * 80)
# print("Passed All Tests!")
# print ("-" * 80)


def plot_embeddings(M_reduced, word2Ind, words):
    """
    可视化词语，画出词嵌入的2d散点图（每个词语的位置）,注意这里画的图只是2维的平面图形
    :param M_reduced: k维的word embeddings矩阵，形状（num_words, k）
    :param word2Ind: 词->索引 表
    :param words: 想要可视化出来的词语
    :return:
    """

    for word in words:
        index = word2Ind[word]
        embedding = M_reduced[index]
        x, y = embedding[0], embedding[1]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x, y, word, fontsize=9)
    plt.show()
# reuters_corpus = read_corpus()
# M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(reuters_corpus)
# M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
# np.linalg.norm：求范数，axis=1，求行的范数
# M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
# print(M_lengths)
# print(M_lengths[:, np.newaxis])
# np.newaxis 广播？通俗讲，在这个位置增加一个维度 https://blog.csdn.net/THMAIL/article/details/121762644
# 目的是为了控制形状也为num_words * k, numpy才能做除法，对于这个式子的目的就是，让word embedding 除以它的范数，得到的结果范数为1，数据变得更大一些，好画图
# M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis]
# print(M_normalized)
# words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
# plot_embeddings(M_normalized, word2Ind_co_occurrence, words)


def load_word2vec():
    """
    加载word2vec，这是一个基于预测的词向量
    :return: 总共300万词的嵌入层，每个维度为300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.index_to_key)
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin
# wv_from_bin = load_word2vec()


def get_matrix_of_vectors(wv_from_bin, required_words=['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']):
    """
    降低词嵌入/词向量的维度
    :param wv_from_bin: word2vec: 总300万的嵌入层
    :param required_words:
    :return:
        - M: 形状为（num_words， k）的矩阵
        - 对应词语到M矩阵中行数的字典表
    """

    import random
    words = list(wv_from_bin.index_to_key)
    print("Shuffling words ...")
    random.shuffle(words)
    words = words[:10000]
    print("Putting %i words into word2Ind and matrix M..." % len(words))
    word2Ind = {}
    M = []
    curInd = 0
    for w in words:
        try:
            M.append(wv_from_bin.get_vector(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    for w in required_words:
        try:
            M.append(wv_from_bin.get_vector(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    M = np.stack(M)
    print("Done.")
    return M, word2Ind
# M, word2Ind = get_matrix_of_vectors(wv_from_bin)
# M_reduced = reduce_to_k_dim(M, k=2)
# print("M", M)
# print("M_reduced", M_reduced)
# words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
# plot_embeddings(M_reduced, word2Ind, words)
#
#
# print("wv_from_bin.most_similar(once)",wv_from_bin.most_similar("once"))
#
# w1 = "men"
# w2 = "gentlemen"
# w3 = "women"
# w1_w2_dist = wv_from_bin.distance(w1, w2)
# w1_w3_dist = wv_from_bin.distance(w1, w3)
#
# print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
# print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))
# print()
# pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'king'], negative=['man']))
# print()
# pprint.pprint(wv_from_bin.most_similar(positive=['winter', 'hot'], negative=['summer']))
# print()
# pprint.pprint(wv_from_bin.most_similar(positive=['grandson', 'old'], negative=['grandfather']))
# print()
# pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'boss'], negative=['man']))
# print()
# pprint.pprint(wv_from_bin.most_similar(positive=['man', 'boss'], negative=['woman']))
#
#
# np.warnings.filterwarnings('ignore')
# print("black:waving :: white: ?")
# pprint.pprint(wv_from_bin.most_similar(positive=['white', 'waving'], negative=['black']))
# print("white:waving :: black: ?")
# pprint.pprint(wv_from_bin.most_similar(positive=['black', 'waving'], negative=['white']))
