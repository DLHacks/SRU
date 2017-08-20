from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sru import SRU

torch.cuda.set_device(2)


''' データセット準備 '''

def set_data():
    mnist = fetch_mldata('MNIST original')
    mnist_X, mnist_y = shuffle(mnist.data, mnist.target, random_state=42)
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
    mnist_X = flatten_img(mnist_X)


    # X.shape => (n_samples, seq_len, n_features) に変換
    mnist_X = mnist_X[:, :, np.newaxis]


    # 訓練、テスト、検証データに分割
    train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y,
                                                        test_size=0.2,
                                                        random_state=42)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y,
                                                          test_size=0.1,
                                                          random_state=42)

    return train_X, test_X, train_y, test_y, valid_X, valid_y

train_X, test_X, train_y, test_y, valid_X, valid_y = set_data()


''' 訓練の準備 '''

import time
import math
import torch.optim as optim

# 計算時間を表示させる
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# mini-batchあたりの訓練
def train(model, inputs, labels, optimizer, criterion, clip):
    # 隠れ変数の初期化
    model.initHidden(inputs.size(1))
    # 勾配の初期化
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), clip)
    optimizer.step()
    return loss.data[0]


# 検証
def validate(model, inputs, labels, optimizer, criterion):
    # 隠れ変数の初期化
    model.initHidden(inputs.size(1))
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    return loss.data[0]


''' パラメータの準備 '''

from hyperopt import fmin, tpe, hp, rand

parameter_space = {
	'phi_size':hp.quniform('phi_size', 1, 256, q=1),
    'r_size':hp.quniform('num_layers', 1, 100, q=1),
	'l_rate': hp.loguniform("l_rate", -10, 0),
    'clip': hp.loguniform("lr", 0, 3)
}



''' 目的関数の定義 '''

def objective(args):
    print(args)
    phi_size = int(args['phi_size'])
    r_size   = int(args['r_size'])
    lr       = args['l_rate']
    clip     = args['clip'] 

    train_X, test_X, train_y, test_y, valid_X, valid_y = set_data()
    input_size = train_X.shape[2]
    output_size = np.unique(train_y).size

    # インスタンスの作成
    model = SRU(input_size, phi_size, r_size, output_size)
    model.cuda()
    model.initWeight()

    # loss, optimizerの定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    n_epochs = 8
    batch_size = 256
    n_batches = train_X.shape[0]//batch_size
    n_batches_v = valid_X.shape[0]//batch_size
    start_time = time.time()

    for epoch in range(n_epochs):
        train_cost, valid_cost = 0, 0

        train_X, train_y = shuffle(train_X, train_y, random_state=42)

        # 訓練
        model.train()
        train_X_t = np.transpose(train_X, (1, 0, 2)) # X.shape => (seq_len, n_samples, n_features) に変換
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            inputs, labels = train_X_t[:, start:end, :], train_y[start:end]
            inputs, labels = Variable(torch.from_numpy(inputs).cuda()
                             ), Variable(torch.from_numpy(labels).cuda())
            train_cost += train(model, inputs, labels, optimizer, criterion, clip) / n_batches

        # 検証
        model.eval()
        valid_X_t = np.transpose(valid_X, (1, 0, 2)) # X.shape => (seq_len, n_samples, n_features) に変換
        for i in range(n_batches_v):
            start = i * batch_size
            end = start + batch_size
            inputs, labels = valid_X_t[:, start:end, :], valid_y[start:end]
            inputs, labels = Variable(torch.from_numpy(inputs).cuda()
                             ), Variable(torch.from_numpy(labels).cuda())
            valid_cost += validate(model, inputs, labels, optimizer, criterion) / n_batches_v

        print('EPOCH:: %i, (%s) Training cost: %.5f, Validation cost: %.5f' % (epoch + 1,
                           timeSince(start_time), train_cost, valid_cost))
        
    return valid_cost

best = fmin(objective, parameter_space, algo=rand.suggest, max_evals=100)
print(best)