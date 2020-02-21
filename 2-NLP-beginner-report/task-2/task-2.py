#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from load_data import *
from Word2Vec import *

data_path = "data"
n_class = 5

if __name__ == "__main__":
    np.seterr(divide='ignore',invalid='ignore') #RuntimeWarning: invalid value encountered in true_divide
    train_data = pd.read_csv(os.path.join(data_path, 'train.tsv'), sep='\t')
    test_data = pd.read_csv(os.path.join(data_path, 'test.tsv'), sep='\t')

    train_x = np.array(clean_sentence(train_data['Phrase'])) #数据清洗
    train_y = np.array(train_data['Sentiment'].values)
    train_x, dev_x, test_x, train_y, dev_y, test_y = train_test_split(train_x, train_y, train_size=0.8)  #划分训练、验证、测试集

    corpus = [
        'he is a king',
        'she is a queen',
        'he is a man',
        'she is a woman',
        'warsaw is poland capital',
        'berlin is germany capital',
        'paris is france capital',
    ]
    matrixs = getWord2Vec(corpus);

    # for i in matrixs:
    #     print(i.shape)
    print(matrixs)

    # model = CNN(1, n_class)
    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)#创建优化器SGD
    # criterion = nn.CrossEntropyLoss()   #损失函数

    # model = CNN(1, n_class)
    # use_gpu = torch.cuda.is_available()
    # if use_gpu:
    #     model.cuda()

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001)

    # for i in range(10):
    #     y_pred = model(train_x)

    #     loss = criterion(y_pred, train_y)

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    # # testing
    # predict_test_Y = sr.predict(test_x)
    # print("Test Acc %.3f" % ((predict_test_Y == test_y).sum() / len(test_y)))