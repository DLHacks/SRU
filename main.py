# -*- coding: utf-8 -*-
""" Script for implementing hyperopt with rnn models. """

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
from hyperopt import fmin, tpe, hp
from models import SRU, GRU, LSTM


''' コマンドライン引数の設定 '''

parser = argparse.ArgumentParser(description='Run hyperopt')
parser.add_argument('model', type=str, default='sru',
                     help='[sru, gru, lstm]: select your model')
parser.add_argument('--gpu', type=int, default=1,
                     help='set -1 when you use cpu')
parser.add_argument('--iter', type=int, default=30,
                     help='select num of hyperopt iteration')
parser.add_argument('--epochs-per-iter', type=int, default=50,
                     help='select num of epochs for each iteration')
parser.add_argument('--batch-size', type=int, default=512,
                     help='set batch_size, default: 512')
parser.add_argument('--seed', type=int, default=0,
                     help='set random seed')
parser.add_argument('--devise-id', type=int, default=0,
                     help='select gpu devise id')
parser.add_argument('--mode', type=str, default='limited',
                     help='[limited, full]: choose whether you train full or limited parameters')
args       = parser.parse_args()
model_name = args.model
gpu        = args.gpu
iteration  = args.iter
n_epochs   = args.epochs_per_iter
batch_size = args.batch_size
seed       = args.seed
devise_id  = args.devise_id
mode       = args.mode

if gpu == True:
    torch.cuda.set_device(devise_id)
# nn.initの初期値はtorch.cuda.manual_seedでなくtorch.manual_seedで管理
torch.manual_seed(seed)
dir_path = './trained_models/%s_%s' % (model_name, mode)
count = 0
print('Bayesian optimization of %s starts' % model_name)


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
    model.initHidden(batch_size) # 隠れ変数の初期化
    optimizer.zero_grad() # 勾配の初期化
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    torch.nn.utils.clip_grad_norm(model.parameters(), clip) # gradient clipping
    loss.backward()
    optimizer.step()
    acc = (torch.max(outputs, 1)[1] == labels).sum().data[0] / batch_size
    return loss.data[0], acc

# 検証
def test(model, inputs, labels, optimizer, criterion):
    model.initHidden(batch_size)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    acc = (torch.max(outputs, 1)[1] == labels).sum().data[0] / batch_size
    return loss.data[0], acc

# モデルの保存
def checkpoint(model, optimizer, acc):
    filename = os.path.join(dir_path,
               '%s_count-%d_acc-%d' % (model.__class__.__name__, count, acc))
    # modelの状態保存
    torch.save(model.state_dict(), filename + '.model')
    # optimizerの状態保存
    torch.save(optimizer.state_dict(), filename + '.state')


''' パラメータの準備
mode == limited: 学習率、重み減退、ドロップアウト、Gradient Clippingのみチューニング
mode == full:    上記に加えて、隠れ層のunit数などもチューニング
'''

parameter_space = ({
    'l_rate':       hp.loguniform('l_rate', -10, 0), # 注: 'lr' is depricated
    'weight_decay': hp.loguniform('weight_decay', -12, -4),
    'dropout':      hp.uniform('dropout', 0, 1),
    'clip':         hp.loguniform('clip', 0, 10)
})
if mode == 'full':
    if model_name == 'sru':
        parameter_space.update({
            'phi_size': hp.quniform('phi_size', 1, 256, q=1),
            'r_size':   hp.quniform('r_size', 1, 64, q=1),
            'cell_out_size': hp.quniform('cell_out_size', 1, 256, q=1)
        })
    elif model_name in ['gru', 'lstm']:
        parameter_space.update({
            'hidden_size': hp.quniform('hidden_size', 1, 256, q=1),
            'num_layers': hp.quniform('num_layers', 1, 5, q=1),
            'init_forget_bias': hp.loguniform('init_forget_bias', -3, 3)
        })


''' 目的関数の定義 '''

def objective(args):
    global count
    count += 1
    print('-------------------------------------------------------------------')
    print('%d回目' % count)
    print(args)

    lr            = args['l_rate']
    weight_decay  = args['weight_decay']
    dropout       = args['dropout']
    clip          = args['clip']
    if mode == 'full':
        if model_name == 'sru':
            phi_size      = int(args['phi_size'])
            r_size        = int(args['r_size'])
            cell_out_size = int(args['cell_out_size'])
        elif model_name in ['gru', 'lstm']:
            hidden_size = int(args['hidden_size'])
            num_layers  = int(args['num_layers'])
            init_forget_bias = args['init_forget_bias']
    elif mode == 'limited':
        if model_name == 'sru':
            phi_size      = 200
            r_size        = 60
            cell_out_size = 200
        elif model_name in ['gru', 'lstm']:
            hidden_size = 200
            num_layers  = 1
            init_forget_bias = 1

    train_X, test_X, train_y, test_y = load_mnist()
    input_size = train_X.shape[2]
    output_size = np.unique(train_y).size

    # モデルのインスタンスの作成
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
    n_batches = train_X.shape[0]//batch_size
    n_batches_test = test_X.shape[0]//batch_size
    all_cost, all_acc = [], []
    start_time = time.time()
    stop_count = 0

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
            cost, accuracy = test(model, inputs, labels, optimizer, criterion)
            test_cost += cost / n_batches_test
            test_acc += accuracy / n_batches_test

        print('EPOCH:: %i, (%s) train_cost: %.3f, test_cost: %.3f, train_acc: %.3f, test_acc: %.3f' % (epoch + 1,
                           timeSince(start_time), train_cost, test_cost, train_acc, test_acc))

        # costが爆発したときに学習打ち切り
        if test_cost != test_cost or test_cost > 100000:
            print('Stop learning due to the extremely high cost')
            all_acc.append(test_acc)
            break

        # 5epochs連続でtest_costの減少が見られないとき早期打ち切り
        if len(all_cost) > 0 and test_cost >= all_cost[-1]:
            stop_count += 1
        else:
            stop_count = 0
        if stop_count == 5:
            print('Early stopping observing no learning')
            all_acc.append(test_acc)
            break

        # 過去のエポックのtest_accを上回った時だけモデルの保存
        if len(all_acc) == 0 or test_acc > max(all_acc):
            checkpoint(model, optimizer, test_acc*10000)

        all_cost.append(test_cost)
        all_acc.append(test_acc)

    print('max test_acc: %.3f' % max(all_acc))

    # test_accの最大値をhyperoptに評価させる
    return max(all_acc)

best = fmin(objective, parameter_space, algo=tpe.suggest, max_evals=iteration,
            rstate=np.random.RandomState(seed))
print(best)

