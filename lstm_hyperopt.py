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
from lstm import LSTM

torch.cuda.set_device(3)

''' データセット準備 '''

def load_mnist():
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
    mnist_X = flatten_img(mnist_X) # X.shape => (n_samples, seq_len) 
    mnist_X = mnist_X[:, :, np.newaxis] # X.shape => (n_samples, seq_len, n_features) 

    # 訓練、テスト、検証データに分割
    train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y,
                                                        test_size=0.2,
                                                        random_state=42)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y,
                                                          test_size=0.1,
                                                          random_state=42)

    return train_X, test_X, train_y, test_y, valid_X, valid_y

train_X, test_X, train_y, test_y, valid_X, valid_y = load_mnist()


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
    # 隠れ変数の初期化
    model.initHidden(batch_size)
    # 勾配の初期化
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    accuracy = (torch.max(outputs, 1)[1] == labels).sum().data[0] / batch_size
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), clip)
    optimizer.step()
    return loss.data[0], accuracy


# 検証
def validate(model, inputs, labels, optimizer, criterion):
    # 隠れ変数の初期化
    batch_size = inputs.size(1)
    model.initHidden(batch_size)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    accuracy = (torch.max(outputs, 1)[1] == labels).sum().data[0] / batch_size
    return loss.data[0], accuracy


''' パラメータの準備 '''

parameter_space = {
	'hidden_size':hp.quniform('hidden_size', 1, 256, q=1),
    'num_layers':hp.quniform('num_layers', 1, 5, q=1),
	'l_rate': hp.loguniform('l_rate', -10, 0),
    'lr_decay':hp.uniform('lr_decay', 0.8, 0.999),
    'init_forget_bias': hp.uniform('init_forget_bias', 0, 20),
    'dropout':hp.uniform('dropout', 0, 1),
    'clip': hp.loguniform('lr', 0, 10)
}


''' 目的関数の定義 '''

def objective(args):
    print(args)
    hidden_size      = int(args['hidden_size'])
    num_layers       = int(args['num_layers'])
    lr               = args['l_rate']
    lr_decay         = args['lr_decay']
    init_forget_bias = args['init_forget_bias']
    dropout          = args['dropout']
    clip             = int(args['clip'])

    torch.cuda.manual_seed(42)
    train_X, test_X, train_y, test_y, valid_X, valid_y = load_mnist()
    input_size = train_X.shape[2]
    output_size = np.unique(train_y).size

    # インスタンスの作成
    model = LSTM(input_size, hidden_size, output_size, num_layers, dropout)
    model.cuda()
    model.initWeight(init_forget_bias)

    # loss, optimizerの定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=lr_decay)

    ''' 訓練 '''
    n_epochs = 8
    batch_size = 128
    n_batches = train_X.shape[0]//batch_size
    n_batches_v = valid_X.shape[0]//batch_size
    all_acc = []
    start_time = time.time()

    for epoch in range(n_epochs):
        train_cost, valid_cost, train_acc, valid_acc  = 0, 0, 0, 0
        train_X, train_y = shuffle(train_X, train_y, random_state=epoch)

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
        valid_X_t = np.transpose(valid_X, (1, 0, 2)) # X.shape => (seq_len, n_samples, n_features) に変換
        for i in range(n_batches_v):
            start = i * batch_size
            end = start + batch_size
            inputs, labels = valid_X_t[:, start:end, :], valid_y[start:end]
            inputs, labels = Variable(torch.from_numpy(inputs).cuda()
                             ), Variable(torch.from_numpy(labels).cuda())
            cost, accuracy = validate(model, inputs, labels, optimizer, criterion)
            valid_cost += cost / n_batches_v
            valid_acc += accuracy / n_batches_v

        all_acc.append(valid_acc)
        print('EPOCH:: %i, (%s) train_cost: %.3f, valid_cost: %.3f, train_acc: %.3f, valid_acc: %.3f' % (epoch + 1,
                           timeSince(start_time), train_cost, valid_cost, train_acc, valid_acc))

    return max(all_acc)

best = fmin(objective, parameter_space, algo=rand.suggest, max_evals=100)
print(best)
