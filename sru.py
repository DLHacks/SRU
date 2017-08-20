import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init


class SRU(nn.Module):
    def __init__(self, input_size, phi_size, r_size, output_size, A=[0, 0.5, 0.9, 0.99], gpu=True):
        """
        input_size:  入力xの特徴量数
        phi_size:    phiのユニット数。\mu^{\alpha}の次元とも等しい
        r_size:      rのユニット数。
        output_size: 出力outputの次元
        A:           [\alpha_1, \alpha_2, ..., \alpha_m]
        """

        super(SRU, self).__init__()

        self._gpu = gpu
        self.n_alpha      = len(A)
        self.phi_size = phi_size
        self.mu_size = self.phi_size * self.n_alpha # muのユニット数 = phiのユニット数 * alphaの個数

        # 各結合の定義
        self.mu2r      = nn.Linear(self.mu_size, r_size)
        self.xr2phi    = nn.Linear(input_size + r_size, phi_size)
        self.mu2o      = nn.Linear(self.mu_size, output_size)
        
        # muphi2phiの準備
        # A_mask: Kronecker product of (A, ones(1, phi_size)),  shape => (1, mu_dim)
        if self._gpu == True:
            self.A_mask = torch.Tensor([x for x in(A) for i in range(phi_size)]).view(1, -1).cuda()
        else:
            self.A_mask = torch.Tensor([x for x in(A) for i in range(phi_size)]).view(1, -1)
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
        outputs = F.relu(self.mu2o(self.mu))
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
        if self._gpu == True:
            self.mu = Variable(torch.zeros(batch_size, self.mu_size)).cuda()
        else:
            self.mu = Variable(torch.zeros(batch_size, self.mu_size))

