# -*- coding:utf-8 -*-
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from gensim import corpora, models
import time
import numpy as np

if __name__ == "__main__":
    begin = time.time()
    corpus = []   # 存储文档
    tokens = []   # 存储文档中的单词
    # 读取文档的操作
    for line in open('./toy.txt','r').readlines():
        if '\xef\xbb\xbf' in line:
            line = line.replace('\xef\xbb\xbf', ' ')
        corpus.append(line.strip())

    # 去标点符号，去截止词的操作
    en_stop = get_stop_words('en')   # 利用Pypi的stop_words包，需要去掉stop_words

    # # 提取主干的词语的操作
    # p_stemmer = PorterStemmer()

    # 分词的操作
    tokenizer = RegexpTokenizer(r'\w+')
    for text in corpus:
        raw = text.lower()
        token = tokenizer.tokenize(raw)
        stop_remove_token = [word for word in token if word not in en_stop]
        tokens.append(stop_remove_token)

    # 得到文档-单词矩阵 （直接利用统计词频得到特征）
    dictionary = corpora.Dictionary(tokens)   # 得到单词的ID,统计单词出现的次数以及统计信息

    texts = [dictionary.doc2bow(text) for text in tokens]    # 将dictionary转化为一个词袋，得到文档-单词矩阵

    # 利用tf-idf来做为特征进行处理
    # texts_tf_idf = models.TfidfModel(texts)[texts]     # 文档的tf-idf形式(训练加转换的模式)

    hdp = models.hdpmodel.HdpModel(corpus=texts, id2word=dictionary)
    lda = models.ldamodel.LdaModel(corpus=texts, id2word=dictionary, num_topics=5, iterations=1000)
    perplex = lda.log_perplexity(texts, total_docs=5)

    end = time.time()
    print("Finish in ", end-begin)

    for topic in lda.print_topics(num_topics=5, num_words=10):
        print(topic,)
    print("Per word perplexity: {:.3f}".format(np.exp2(-perplex)))
