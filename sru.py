import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init


class SRU(nn.Module):
    def __init__(self, input_size, phi_size, r_size, output_size, A=[0, 0.5, 0.9, 0.99], gpu=True):
        """
        input_size:   入力xの特徴量数
        phi_size:     phiのユニット数。\mu^{\alpha}の次元とも等しい
        r_size:       rのユニット数。
        output_size:  出力outputの次元
        A:            [\alpha_1, \alpha_2, ..., \alpha_m]
        """

        super(SRU, self).__init__()

        self._gpu = gpu
        self.n_alpha      = len(A)
        # Aをリストからtensorに変換（後の計算のため）
        if self._gpu == True:
            self.A       = torch.Tensor(A).view(1, -1).cuda()
        else:
            self.A       = torch.Tensor(A).view(1, -1)
        self.phi_size = phi_size
        # muのユニット数 = phiのユニット数 * alphaの個数
        self.mu_size = self.phi_size * self.n_alpha

        # 各結合の定義
        self.mu2r    = nn.Linear(self.mu_size, r_size)
        self.xr2phi  = nn.Linear(input_size + r_size, phi_size)
        self.mu2o    = nn.Linear(self.mu_size, output_size)

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
        すべての\alphaについて、\mu_t^{(\alpha)} = \alpha \mu_{t-1}^{(\alpha)} + (1-\alpha) \phi_t を同時に行う
            A_mask:   Kronecker product of (A, ones(1, phi_size)),   shape => (1, mu_dim)
            phi_tile: Kronecker product of (ones(1, n_alpha), phi), shape => (sample_size, mu_dim)
        '''
        if self._gpu:
            A_mask = kronecker_product(self.A, torch.ones(1, self.phi_size).cuda())
            phi_tile = kronecker_product(Variable(torch.ones(1, self.n_alpha).cuda()), phi)
        else:
            A_mask = kronecker_product(self.A, torch.ones(1, self.phi_size))
            phi_tile = kronecker_product(Variable(torch.ones(1, self.n_alpha)), phi)

        # 要素積をとるためにA_maskをVariableに変換するが、A_maskは定数項なのでrequires_grad=Falseをつける
        A_mask = Variable(A_mask, requires_grad=False)
        mu = torch.mul(A_mask, mu) + torch.mul((1-A_mask), phi_tile)
        return mu

    def initWeight(self):
        for name, params in self.named_parameters():
            # weightをxavierで初期化
            if 'weight' in name:
                init.xavier_uniform(params)
            # それ以外のbiasを0に初期化
            else:
                init.constant(params, 0)

    def initHidden(self, batch_size):
        if self._gpu == True:
            self.mu = Variable(torch.zeros(batch_size, self.mu_size)).cuda()
        else:
            self.mu = Variable(torch.zeros(batch_size, self.mu_size))


def kronecker_product(t1, t2):
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2



