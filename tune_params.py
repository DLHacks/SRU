""" Script for implementing hyperopt with rnn models. """

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
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from hyperopt import fmin, tpe, hp, rand
from models import SRU, GRU, LSTM


''' コマンドライン引数の設定 '''

parser = argparse.ArgumentParser(description='Run hyperopt')
parser.add_argument('--model', type=str, default='sru',
                     help='[sru, gru, lstm]: select your model')
parser.add_argument('--each-epoch', type=int, default=50,
                     help='select num of epochs for each trial')
parser.add_argument('--seed', type=int, default=0,
                     help='set random seed')
parser.add_argument('--devise-id', type=int, default=0,
                     help='select gpu devise id')
parser.add_argument('--mode', type=str, default='limited',
                     help='[limited, full]: choose whether you train full or limited parameters')
args       = parser.parse_args()
model_name = args.model
n_epochs   = args.each_epoch
seed       = args.seed
devise_id  = args.devise_id
mode       = args.mode

torch.cuda.manual_seed(seed)
torch.cuda.set_device(devise_id)
print('Bayesian optimization of %s' % model_name)

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
    batch_size = inputs.size(1)
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
def testate(model, inputs, labels, optimizer, criterion):
    batch_size = inputs.size(1)
    model.initHidden(batch_size)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    acc = (torch.max(outputs, 1)[1] == labels).sum().data[0] / batch_size
    return loss.data[0], acc


''' パラメータの準備 '''

parameter_space = ({
    'l_rate': hp.loguniform('l_rate', -10, 0),
    'lr_decay': hp.uniform('lr_decay', 0.8, 0.999),
    'weight_decay': hp.loguniform('weight_decay', -12, -6),
    'dropout':hp.uniform('dropout', 0, 1),
    'clip': hp.loguniform('clip', 0, 10)
})
if mode == 'full':
    if model_name == 'sru':
        parameter_space.update({
            'phi_size': hp.quniform('phi_size', 1, 256, q=1),
            'r_size': hp.quniform('r_size', 1, 64, q=1),
            'cell_out_size': hp.quniform('cell_out_size', 1, 256, q=1)
        })
    elif model_name in ['gru', 'lstm']:
        parameter_space.update({
            'hidden_size': hp.quniform('hidden_size', 1, 256, q=1),
            'num_layers': hp.quniform('num_layers', 1, 5, q=1),
            'init_forget_bias': hp.loguniform('init_forget_bias', -3, 3)
        })
count = 0


''' 目的関数の定義 '''

def objective(args):
    global count
    count += 1

    print(args)
    lr            = args['l_rate']
    lr_decay      = args['lr_decay']
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
    elif: mode == 'limited'
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
        model = SRU(input_size, phi_size, r_size, cell_out_size, output_size, dropout=dropout)
        model.initWeight()
    elif model_name == 'gru':
        model = GRU(input_size, hidden_size, output_size, num_layers, dropout)
        model.initWeight(init_forget_bias)
    elif model_name == 'lstm':
        model = LSTM(input_size, hidden_size, output_size, num_layers, dropout)
        model.initWeight(init_forget_bias)
    model.cuda()

    # loss, optimizerの定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=100, gamma=lr_decay)


    ''' 訓練 '''
    batch_size = 128
    n_batches = train_X.shape[0]//batch_size
    n_batches_test = test_X.shape[0]//batch_size
    all_acc = []
    start_time = time.time()

    for epoch in range(n_epochs):
        train_cost, test_cost, train_acc, test_acc  = 0, 0, 0, 0
        train_X, train_y = shuffle(train_X, train_y, random_state=seed+epoch)

        # 訓練
        model.train()
        train_X_t = np.transpose(train_X, (1, 0, 2)) # X.shape => (seq_len, n_samples, n_features) に変換
        for i in range(n_batches):
            scheduler.step()
            start = i * batch_size
            end = start + batch_size
            inputs, labels = train_X_t[:, start:end, :], train_y[start:end]
            inputs, labels = Variable(torch.from_numpy(inputs).cuda()
                             ), Variable(torch.from_numpy(labels).cuda())
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
            inputs, labels = Variable(torch.from_numpy(inputs).cuda()
                             ), Variable(torch.from_numpy(labels).cuda())
            cost, accuracy = testate(model, inputs, labels, optimizer, criterion)
            test_cost += cost / n_batches_test
            test_acc += accuracy / n_batches_test

        all_acc.append(test_acc)
        print('EPOCH:: %i, (%s) train_cost: %.3f, test_cost: %.3f, train_acc: %.3f, test_acc: %.3f' % (epoch + 1,
                           timeSince(start_time), train_cost, test_cost, train_acc, test_acc))

    print('%d回目: %.3' % (count, max(all_acc)))
    # test_accの最大値を返す
    return max(all_acc)

best = fmin(objective, parameter_space, algo=rand.suggest, max_evals=100)
print(best)

