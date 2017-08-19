import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        

    def forward(self, inputs):
        # hidden = (h_t, c_t)
        _, self.hidden = self.lstm(inputs, self.hidden)
        # extract the last hidden layer from h_t(n_layers, n_samples, hidden_size)
        htL = self.hidden[0][-1]
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

    def initHidden(self, batch_size, gpu=True):
        if gpu == True:
            self.hidden = (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()))
        else:
            self.hidden = (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)),
                           Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)))

