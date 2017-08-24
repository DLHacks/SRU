import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init


class SRU(nn.Module):
    def __init__(self, input_size, phi_size, r_size, cell_out_size, output_size, A=[0, 0.5, 0.9, 0.99, 0.999], dropout=0, gpu=True):
        """
        input_size:    inputの特徴量数
        phi_size:      phiのユニット数。\mu^{\alpha}の次元とも等しい
        r_size:        rのユニット数
        cell_out_size: SRUCellからの出力のunit数
        output_size:   outputの次元
        A:             [\alpha_1, \alpha_2, ..., \alpha_m]
        """

        super(SRU, self).__init__()

        self._gpu = gpu
        self.n_alpha  = len(A)
        self.phi_size = phi_size
        self.mu_size  = self.phi_size * self.n_alpha # muのユニット数 = phiのユニット数 * alphaの個数

        # 各結合の定義
        self.mu2r   = nn.Linear(self.mu_size, r_size)
        self.xr2phi = nn.Linear(input_size + r_size, phi_size)
        self.mu2o   = nn.Linear(self.mu_size, cell_out_size)
        self.drop   = nn.Dropout(p=dropout)
        self.linear = nn.Linear(cell_out_size, output_size)

        # muphi2phiの準備
        # A_mask: Kronecker product of (A, ones(1, phi_size)),  shape => (1, mu_dim)
        self.A_mask = torch.Tensor([x for x in(A) for i in range(phi_size)]).view(1, -1)
        if self._gpu == True:
            self.A_mask = self.A_mask.cuda()
        # A_maskは定数項なのでrequires_grad=Falseをつける
        self.A_mask = Variable(self.A_mask, requires_grad=False)

    def forward(self, inputs):
        '''
        inputs.size()  => (seq_len, sample_size, x_dim)
        mu.size() => (sample_size, mu_dim)
        '''
        for x in inputs:
            r = F.relu(self.mu2r(self.mu))
            phi = F.relu(self.xr2phi(torch.cat((x, r), 1)))
            self.mu = self.muphi2mu(self.mu, phi)
        cell_out = F.relu(self.mu2o(self.mu))
        cell_out = self.drop(cell_out)
        outputs  = self.linear(cell_out)
        return outputs

    def muphi2mu(self, mu, phi):
        '''
        数式: \mu = A_mask * \mu + (1-A_mask) * phi_tile
            A_mask:   Kronecker product of (A, ones(1, phi_size)),  shape => (1, mu_dim)
            phi_tile: Kronecker product of (ones(1, n_alpha), phi), shape => (sample_size, mu_dim)
        '''
        phi_tile = phi.repeat(1, self.n_alpha)
        mu = torch.mul(self.A_mask, mu) + torch.mul((1-self.A_mask), phi_tile)
        return mu

    def initWeight(self):
        for name, params in self.named_parameters():
            # weightをxavierで初期化
            if 'weight' in name:
                init.xavier_uniform(params, init.calculate_gain('relu'))
            # biasを0で初期化
            else:
                init.constant(params, 0)

    def initHidden(self, batch_size):
        self.mu = Variable(torch.zeros(batch_size, self.mu_size))
        if self._gpu == True:
            self.mu = self.mu.cuda()


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0, gpu=True):
        super(GRU, self).__init__()

        self._gpu        = gpu
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # 各layerの定義
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers)
        self.drop = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        _, self.hidden = self.gru(inputs, self.hidden)
        # extract the last hidden layer from ht(n_layers, n_samples, hidden_size)
        htL = self.hidden[-1]
        htL = self.drop(htL)
        outputs = self.linear(htL)
        return outputs

    def initWeight(self, init_forget_bias=1):
        # See details in https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
        for name, params in self.named_parameters():
            # weightをxavierで初期化
            if 'weight' in name:
                init.xavier_uniform(params)

            # 忘却しやすくなるようGRUのb_iz, b_hzを初期化
            elif 'gru.bias_ih_l' in name:
                b_ir, b_iz, b_in = params.chunk(3, 0)
                init.constant(b_iz, init_forget_bias)
            elif 'gru.bias_hh_l' in name:
                b_hr, b_hz, b_hn = params.chunk(3, 0)
                init.constant(b_hz, init_forget_bias)

            # それ以外のbiasを0に初期化
            else:
                init.constant(params, 0)

    def initHidden(self, batch_size):
        self.hidden = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        if self._gpu == True:
            self.hidden = self.hidden.cuda()


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0, gpu=True):
        super(LSTM, self).__init__()

        self._gpu        = gpu
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # 各layerの定義
        self.lstm   = nn.LSTM(input_size, hidden_size, num_layers)
        self.drop   = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        # hidden = (h_t, c_t)
        _, self.hidden = self.lstm(inputs, self.hidden)
        # extract the last hidden layer from h_t(n_layers, n_samples, hidden_size)
        htL = self.hidden[0][-1]
        htL = self.drop(htL)
        outputs = self.linear(htL)
        return outputs

    def initWeight(self, init_forget_bias=1):
        # See https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
        for name, params in self.named_parameters():
            # weightをxavierで初期化
            if 'weight' in name:
                init.xavier_uniform(params)

            # 忘却しやすくなるようLSTMのb_if, b_hfを初期化
            elif 'lstm.bias_ih_l' in name:
                b_ii, b_if, b_ig, b_i0 = params.chunk(4, 0)
                init.constant(b_if, init_forget_bias)
            elif 'lstm.bias_hh_l' in name:
                b_hi, b_hf, b_hg, b_h0 = params.chunk(4, 0)
                init.constant(b_hf, init_forget_bias)

            # それ以外のbiasを0に初期化
            else:
                init.constant(params, 0)

    def initHidden(self, batch_size):
        if self._gpu == True:
            self.hidden = (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()))
        else:
            self.hidden = (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)),
                           Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)))
