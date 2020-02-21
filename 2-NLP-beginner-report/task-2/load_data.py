#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import re
from gensim.models.word2vec import Word2Vec

#数据清洗
def clean_sentence(df):
    result = []
    for sentence in df:
        result_text = re.sub("[^a-zA-Z]"," ", sentence)
        words = result_text.lower()
        result.append(words)
    return result

#训练集、验证集、测试集划分
def train_test_split(x, y, train_size, shuffle = True):
    assert x.shape[0] == y.shape[0]
    len = x.shape[0]
    index = np.arange(0, len)
    if shuffle:
        np.random.shuffle(index)
    train_num = int(train_size * len)
    dev_num = int((len - train_num)/2)
    test_num = len - train_num - dev_num
    return x[index[:train_num]], x[index[train_num: train_num+dev_num]], x[index[-test_num:]], \
        y[index[:train_num]], y[index[train_num:train_num + dev_num]], y[index[-test_num:]]

