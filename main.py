# -*- coding: utf-8 -*-
""" Script for running RNNs with fixed parameters. """

import os
import argparse
import time
import math
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models import SRU, GRU, LSTM

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, default='sru',
                     help='[sru, gru, lstm]: select your model')
parser.add_argument('--gpu', type=int, default=1,
                     help='set -1 when you use cpu')
parser.add_argument('--batch-size', type=int, default=512,
                     help='set batch_size, default: 512')
parser.add_argument('--seed', type=int, default=0,
                     help='set random seed')
parser.add_argument('--devise-id', type=int, default=0,
                     help='select gpu devise id')

args       = parser.parse_args()
model_name = args.model
gpu        = args.gpu
batch_size = args.batch_size
seed       = args.seed
devise_id  = args.devise_id

torch.cuda.set_device(devise_id)
torch.manual_seed(seed)
dir_path = './trained_models/main'
print('%s starting......' % model_name)


''' データセット準備 '''

def load_mnist():
    mnist = fetch_mldata('MNIST original')
    mnist_X, mnist_y = shuffle(mnist.data, mnist.target, random_state=seed)
    mnist_X = mnist_X / 255.0

    # pytorch用に型変換
    mnist_X, mnist_y = mnist_X.astype('float32'), mnist_y.astype('int64')

    # 2次元の画像を、各行を互い違いにして1次元に変換
    def flatten_img(images):
        '''
        images: shape => (n, rows, columns)
        output: shape => (n, rows*columns)
        '''
        n_rows    = images.shape[1]
        n_columns = images.shape[2]
        for num in range(n_rows):
            if num % 2 != 0:
                images[:, num, :] = images[:, num, :][:, ::-1]
        output = images.reshape(-1, n_rows*n_columns)
        return output

    mnist_X = mnist_X.reshape(-1, 28, 28)
    mnist_X = flatten_img(mnist_X) # X.shape => (n_samples, seq_len)
    mnist_X = mnist_X[:, :, np.newaxis] # X.shape => (n_samples, seq_len, n_features)

    # 訓練、テストデータに分割
    train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y,
                                                        test_size=0.2,
                                                        random_state=seed)
    return train_X, test_X, train_y, test_y

train_X, test_X, train_y, test_y = load_mnist()


''' 訓練の準備 '''

# 計算時間を表示させる
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# batchあたりの訓練
def train(model, inputs, labels, optimizer, criterion, clip):
    batch_size = inputs.size(1)
    model.initHidden(batch_size) # 隠れ変数の初期化
    optimizer.zero_grad() # 勾配の初期化
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    torch.nn.utils.clip_grad_norm(model.parameters(), clip) # gradient clipping
    loss.backward()
    optimizer.step()
    acc = (torch.max(outputs, 1)[1] == labels).float().sum().data[0] / batch_size
    return loss.data[0], acc

# 検証
def test(model, inputs, labels, criterion):
    batch_size = inputs.size(1)
    model.initHidden(batch_size)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    acc = (torch.max(outputs, 1)[1] == labels).float().sum().data[0] / batch_size
    return loss.data[0], acc

# モデルの保存
def checkpoint(model, optimizer, acc):
    filename = os.path.join(dir_path, '%s_acc-%d' % (model.__class__.__name__, acc))
    # modelの状態保存
    torch.save(model.state_dict(), filename + '.model')
    # optimizerの状態保存
    torch.save(optimizer.state_dict(), filename + '.state')


input_size = train_X.shape[2]
output_size = np.unique(train_y).size

# パラメータの設定
if model_name == 'sru':
    phi_size      = 200
    r_size        = 60
    cell_out_size = 200
    lr = 0.0005174277555790016
    weight_decay = 3.9473232493735065e-05
    dropout = 0.7281811891811246
    clip = 17.380962431598327
elif model_name =='gru':
    hidden_size = 200
    num_layers  = 1
    init_forget_bias = 1
    lr = 0.0037046604805510137
    weight_decay = 0.00011813244108811544
    dropout = 0.26173877481275953
    clip = 2925.4042227640757
elif model_name == 'lstm':
    hidden_size = 200
    num_layers  = 1
    init_forget_bias = 1
    lr = 0.00016654418947982137
    weight_decay = 7.040822706204121e-05
    dropout = 0.18404592540409914
    clip = 4389.748805208904

# モデルのインスタンス作成
if model_name == 'sru':
    model = SRU(input_size, phi_size, r_size, cell_out_size, output_size, dropout=dropout, gpu=gpu)
    model.initWeight()
elif model_name == 'gru':
    model = GRU(input_size, hidden_size, output_size, num_layers, dropout, gpu=gpu)
    model.initWeight(init_forget_bias)
elif model_name == 'lstm':
    model = LSTM(input_size, hidden_size, output_size, num_layers, dropout, gpu=gpu)
    model.initWeight(init_forget_bias)
if gpu == True:
    model.cuda()

# loss, optimizerの定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


''' 訓練 '''

n_epochs = 400
n_batches = train_X.shape[0]//batch_size
n_batches_test = test_X.shape[0]//batch_size
all_acc = []
start_time = time.time()

for epoch in range(n_epochs):
    train_cost, test_cost, train_acc, test_acc  = 0, 0, 0, 0
    train_X, train_y = shuffle(train_X, train_y, random_state=seed)

    # 訓練
    model.train()
    train_X_t = np.transpose(train_X, (1, 0, 2)) # X.shape => (seq_len, n_samples, n_features) に変換
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        inputs, labels = train_X_t[:, start:end, :], train_y[start:end]
        inputs, labels = Variable(torch.from_numpy(inputs)
                         ), Variable(torch.from_numpy(labels))
        if gpu == True:
            inputs, labels = inputs.cuda(), labels.cuda()
        cost, accuracy = train(model, inputs, labels, optimizer, criterion, clip)
        train_cost += cost / n_batches
        train_acc  += accuracy / n_batches

    # 検証
    model.eval()
    test_X_t = np.transpose(test_X, (1, 0, 2))
    for i in range(n_batches_test):
        start = i * batch_size
        end = start + batch_size
        inputs, labels = test_X_t[:, start:end, :], test_y[start:end]
        inputs, labels = Variable(torch.from_numpy(inputs)
                         ), Variable(torch.from_numpy(labels))
        if gpu == True:
            inputs, labels = inputs.cuda(), labels.cuda()
        cost, accuracy = test(model, inputs, labels, criterion)
        test_cost += cost / n_batches_test
        test_acc += accuracy / n_batches_test

    print('EPOCH:: %i, (%s) train_cost: %.3f, test_cost: %.3f, train_acc: %.3f, test_acc: %.3f' % (epoch + 1,
                       timeSince(start_time), train_cost, test_cost, train_acc, test_acc))

    # 過去のエポックのtest_accを上回った時だけモデルの保存
    if len(all_acc) == 0 or test_acc > max(all_acc):
        checkpoint(model, optimizer, test_acc*10000)
    all_acc.append(test_acc)

print('Finished Training')
