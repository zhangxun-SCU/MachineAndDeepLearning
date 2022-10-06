import pickle
import numpy as np
import os
import random


class Sentiment:
    def __int__(self, path=None, tableSize=1000000):
        if not path:
            path = "utils/datasets/stanfordSentimentTreebank"

        self.path = path
        self.tableSize = tableSize

    def tokens(self):
        if hasattr(self, "_tokens") and self._tokens:
            # 如果对象有_tokens直接返回即可
            return self._tokens

        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        revtokens = []
        idx = 0

        for sentence in self.sentences():
            for w in sentence:
                wordcount += 1
                if not w in tokens:
                    tokens[w] = idx
                    revtokens += [w]
                    tokenfreq[w] = 1
                    idx += 1
                else:
                    tokenfreq[w] += 1

        tokens["UNK"] = idx
        revtokens += ["UNK"]
        tokenfreq["UNK"] = 1
        wordcount += 1

        self._tokens = tokens
        self._tokenfreq = tokenfreq
        self._wordcount = wordcount
        self._revtokens = revtokens
        return self._tokens

    def sentences(self):
        """
        将数据集中的句子化为单词列表，_cunsentlen是到每一行的总共长度
        :return: list of lists, _sentences是将datasetSentences里的句子，化为一个个小写单词，一句话一个list
        """
        if hasattr(self, "_sentences") and self._sentences:
            return self._sentences

        sentences = []
        with open(self.path + "/datasetSentences.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    # 数据集第一行为介绍不要
                    first = False
                    continue

                splitted = line.strip().split()[1:]  # 索引0为编号
                # 处理大小写等问题
                sentences += [[w.lower() for w in splitted]]

        self._sentences = sentences
        self._sentlengths = np.array([len(s) for s in sentences])
        self._cunsentlen = np.cumsum(self._sentlengths)

        return self._sentences

    def numSentences(self):
        """
        求数据集句子条数
        :return: 返回数据集中语句的条数
        """
        if hasattr(self, "_numSentences") and self._numSentences:
            return self._numSentences
        self._numSentences = len(self.sentences())
        return self._numSentences

    def allSentences(self):
        if hasattr(self, "_allSentences") and self._allSentences:
            return self._allSentences

        sentences = self.sentences()
        rejectProb = self.rejectProb()
        tokens = self.tokens()
        allsentences = [[w for w in s
                         if 0 >= rejectProb[tokens[w]] or random.random() >= rejectProb[tokens[w]]]
                        for s in sentences * 30]

        allsentences = [s for s in allsentences if len(s) > 1]

        self._allsentences = allsentences

        return self._allsentences

    def getRandomContext(self, C=5):
        """

        :param C:
        :return:
        """
        allsent = self.allSentences()
        sentID = random.randint(0, len(allsent) - 1)
        sent = allsent[sentID]
        wordID = random.randint(0, len(sent) - 1)
        context = sent[max(0, wordID - C):wordID]

        if wordID + 1 < len(sent):
            context += sent[wordID + 1:min(len(sent), wordID + C + 1)]

        centereord = sent[wordID]
        # 去掉中心词？
        context = [w for w in context if w != centereord]

        if len(context) > 0:
            return centereord, context
        else:
            return self.getRandomContext(C)

    def sent_labels(self):
        """
        得到每个句子对应的sentiment
        :return: 返回句子对应的sentiment
        """
        if hasattr(self, "_sent_labels") and self._sent_labels:
            return self._sent_labels

        # dictionary: word2number(将单词转为数字)， phrases:短语/单词个数
        dictionary = dict()
        phrases = 0
        with open(self.path + "/dictionary.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    # line为空->跳过
                    continue
                splitted = line.strip("|")
                dictionary[splitted[0].lower()] = int(splitted[1])
                phrases += 1
        # 获得字典dictionary长度的列表， labels: 对应下标的数据是相应短语的sentiment值
        labels = [0.0] * phrases
        with open(self.path + "/sentiment_labels.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                line = line.strip()
                if not line:
                    continue
                splitted = line.split("|")
                labels[int(splitted[0])] = float(splitted[1])

        #
        sent_labels = [0.0] * self.numSentences()
        sentences = self.sentences()
        for i in range(self.numSentences()):
            # 遍历每一个句子
            sentence = sentences[i]
            # 再将每一个单词连接为完整的句子并替换左右括号
            full_sent = " ".join(sentence).replace('-lrb-', '(').replace('-rrb-', ')')
            # 得到每个句子对应的sentiment
            sent_labels[i] = labels[dictionary[full_sent]]

        self._sent_labels = sent_labels
        return self._sent_labels

    def dataset_split(self):
        """
        获得编号对应的label，列表内的数据表示编号，数据在_split的第几个列表内表示label(0-2)
        :return:
        """
        if hasattr(self, "_split") and self._split:
            return self._split

        # list of 3 lists
        split = [[] for i in range(3)]
        with open(self.path + "/datasetSplit.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue
                splitted = line.strip().split(",")
                split[int(splitted[1]) - 1] += [int(splitted[0]) - 1]

        self._split = split
        return self._split

    def getRandomTrainSentence(self):
        """

        :return:
        """
        split = self.dataset_split()
        sentId = split[0][random.randint(0, len(split[0]) - 1)]
        return self.sentences()[sentId], self.categorify(self.sent_labels()[sentId])

    #########################################

    def categorify(self, label):
        if label <= 0.2:
            return 0
        elif label <= 0.4:
            return 1
        elif label <= 0.6:
            return 2
        elif label <= 0.8:
            return 3
        else:
            return 4

    def getDevSentences(self):
        return self.getSplitSentences(2)

    def getTestSentences(self):
        return self.getSplitSentences(1)

    def getTrainSentences(self):
        return self.getSplitSentences(0)

    def getSplitSentences(self, split=0):
        ds_split = self.dataset_split()
        return [(self.sentences()[i], self.categorify(self.sent_labels()[i])) for i in ds_split[split]]

    def sampleTable(self):
        if hasattr(self, '_sampleTable') and self._sampleTable is not None:
            return self._sampleTable

        nTokens = len(self.tokens())
        samplingFreq = np.zeros((nTokens,))
        self.allSentences()
        i = 0
        for w in range(nTokens):
            w = self._revtokens[i]
            if w in self._tokenfreq:
                freq = 1.0 * self._tokenfreq[w]
                # Reweigh
                freq = freq ** 0.75
            else:
                freq = 0.0
            samplingFreq[i] = freq
            i += 1

        samplingFreq /= np.sum(samplingFreq)
        samplingFreq = np.cumsum(samplingFreq) * self.tablesize

        self._sampleTable = [0] * self.tablesize

        j = 0
        for i in range(self.tablesize):
            while i > samplingFreq[j]:
                j += 1
            self._sampleTable[i] = j

        return self._sampleTable

    def rejectProb(self):
        if hasattr(self, '_rejectProb') and self._rejectProb is not None:
            return self._rejectProb

        threshold = 1e-5 * self._wordcount

        nTokens = len(self.tokens())
        rejectProb = np.zeros((nTokens,))
        for i in range(nTokens):
            w = self._revtokens[i]
            freq = 1.0 * self._tokenfreq[w]
            # Reweigh
            rejectProb[i] = max(0, 1 - np.sqrt(threshold / freq))

        self._rejectProb = rejectProb
        return self._rejectProb

    def sampleTokenIdx(self):
        return self.sampleTable()[random.randint(0, self.tablesize - 1)]
