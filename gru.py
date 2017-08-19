import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        

    def forward(self, inputs):
        _, self.hidden = self.gru(inputs, self.hidden)
        # extract the last hidden layer from ht(n_layers, n_samples, hidden_size)
        htL = self.hidden[-1]
        outputs = self.linear(htL)
        return outputs
    
    def weight_init(self, init_forget_bias):
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

    def init_hidden(self, batch_size, gpu=True):
        if gpu == True:
            self.hidden = Variable(torch.randn(self.num_layers, batch_size, self.hidden_size).cuda())
        else:
            self.hidden = Variable(torch.randn(self.num_layers, batch_size, self.hidden_size))

