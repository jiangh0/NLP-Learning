import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

'''
摘自：https://github.com/graykode/nlp-tutorial/blob/master/1-2.Word2Vec/Word2Vec_Skipgram_Torch(Softmax).ipynb
已完成-待理解
'''

dtype = torch.FloatTensor #默认类型

#原始数据
sentences = [ "i like dog", "i like cat", "i like animal",
              "dog cat animal", "apple cat dog like", "dog fish milk like",
              "dog cat eyes like", "i like apple", "apple i hate",
              "apple i movie book music like", "cat dog hate", "cat dog like"]

#构造字典
word_sequence = ' '.join(sentences).split()
word_list = list(set(word_sequence))
word_dict = {w: i for i, w in enumerate(word_list)}

#Word2Vec Parameter
batch_size = 20     #每个batch内的数量
embedding_size = 2
voc_size = len(word_list)

#生成词对 【中心词，要预测的词】
skip_grams = []
for i in range(1, len(word_sequence)-1):
    target = word_dict[word_sequence[i]]
    context = [word_dict[word_sequence[i-1]], word_dict[word_sequence[i+1]]]

    for w in context:
        skip_grams.append([target, w])

#随机从词对中抽取数量为size的词对
def random_batch(data, size):
    '''
        data：原始数据（即skip_grams）
        size：batch_size
    '''
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(np.eye(voc_size)[data[i][0]])
        random_labels.append(data[i][1])

    return random_inputs, random_labels

#神经网络模型
class Word2Vec(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(-2 * torch.rand(voc_size, embedding_size) + 1).type(dtype)
        self.WT = nn.Parameter(-2 * torch.rand(embedding_size, voc_size) + 1).type(dtype)

    def forward(self, X):
        # X (one-hot): [batch_size, voc_size]
        hidden_layer = torch.matmul(X, self.W)  # hidden_layer : [batch_size, embedding_size]
        output_layer = torch.matmul(hidden_layer, self.WT)  # output_layer : [batch_size, voc_size]
        print(self.W, self.WT, hidden_layer, output_layer)
        return output_layer

model = Word2Vec()
criterion = nn.CrossEntropyLoss()   #loss function
optimizer = optim.Adam(model.parameters(), lr=0.001) #优化算法 Adam

#Training
for epoch in range(5000):
    input_batch, target_batch = random_batch(skip_grams, batch_size)

    input_batch = Variable(torch.Tensor(input_batch))
    target_batch = Variable(torch.LongTensor(target_batch))

    #正向传播，计算loss
    optimizer.zero_grad()   #初始化梯度为0
    output = model(input_batch)     #model.forward()
    loss = criterion(output, target_batch)  #计算损失
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()     #反向传播
    optimizer.step()    #正向

for i, label in enumerate(word_list):
    #
    W, WT = model.parameters()
    x, y = float(W[i][0]), float(W[i][1])
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()
