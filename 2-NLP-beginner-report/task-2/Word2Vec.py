#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

'''
corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',
]
'''
def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens

def one_hot(idx, vocab_size):
    x = torch.zeros((vocab_size, 1)).float()
    x[idx] = 1
    return x

class Word2Vec():
    def __init__(self, embedding_dims, vocab_size, learning_rate):
        self.w1 = Variable(torch.randn(embedding_dims, vocab_size).float(), requires_grad=True)
        self.w2 = Variable(torch.randn(vocab_size, embedding_dims).float(), requires_grad=True)
        self.learning_rate = learning_rate

    def gd(self, x, y):
        z1 = torch.matmul(self.w1, x)
        z2 = torch.matmul(self.w2, z1)
        log_softmax = F.log_softmax(z2, dim=0)
        loss = F.nll_loss(log_softmax.view(1, -1), y)
        loss.backward()
        with torch.no_grad():
            self.w1 -= self.learning_rate * self.w1.grad
            self.w2 -= self.learning_rate * self.w2.grad
            self.w1.grad.zero_()
            self.w2.grad.zero_()
        return loss.item()

def getWord2Vec(corpus, window_size=2):
    tokenized_corpus = tokenize_corpus(corpus)

    vocab = []
    for s in tokenized_corpus:
        for token in s:
            if token not in vocab:
                vocab.append(token)
    word2idx = {w: idx for (idx, w) in enumerate(vocab)}
    # idx2word = {idx: w for (idx, w) in enumerate(vocab)}

    vocab_size = len(vocab)
    idx_pairs = []
    for s in tokenized_corpus:
        indices = [word2idx[word] for word in s]
        for center_pos in range(len(indices)):
            for w in range(-window_size, window_size+1):
                context_pos = center_pos+w
                if context_pos < 0 or context_pos>= len(indices) or context_pos== center_pos:
                    continue
                context_index = indices[context_pos]
                idx_pairs.append((indices[center_pos], context_index))
    idx_pairs = np.array(idx_pairs)

    word2Vec = Word2Vec(5, vocab_size, 0.001)
    for epo in tqdm(range(100)):
        loss_val = 0
        for data, target in idx_pairs:
            x = Variable(one_hot(data, vocab_size)).float()
            y = Variable(torch.from_numpy(one_hot(data, vocab_size)).long())
            loss_val += word2Vec.gd(x, y)
        if epo % 10 == 0:    
            print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')

    matrixs = torch.zeros((vocab_size, 4, 5))
    for i in tqdm(range(vocab_size)):
        indices = [word2idx[word] for word in tokenized_corpus[i]]
        for idx in indices:
            z1 = word2Vec.w1.matmul(one_hot(idx, vocab_size).float())
            z2 = word2Vec.w2.matmul(z1).view(1,-1)
            print(z2)
            matrixs[idx] = F.log_softmax(z2, dim=0)
            print(matrixs[idx])
        # matrixs.append(matrix)

    return matrixs