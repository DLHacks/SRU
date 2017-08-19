import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class SRU(nn.Module):
    def __init__(self, x_dim, phi_dim, r_dim, o_dim, A, GPU=True):
        """ 
        x_dim:   入力xの次元（特徴量数）
        phi_dim: phiの次元。\mu^{\alpha}の次元とも等しい
        r_dim:   rの次元
        o_dim:   出力oの次元
        A:       [\alpha_1, \alpha_2, ..., \alpha_m], shape: (1, m)
        """

        super(SRU, self).__init__()

        self.gpu     = GPU
        n_alpha      = A.size()[1]
        self.n_alpha = n_alpha
        self.A       = A
        self.phi_dim = phi_dim
        # muの次元 = phiの次元*alphaの個数
        mu_dim = phi_dim * n_alpha 
        self.mu_dim = mu_dim
        
        # 各結合の定義
        self.mu2r    = nn.Linear(mu_dim, r_dim)
        self.xr2phi  = nn.Linear(x_dim + r_dim, phi_dim)
        self.mu2o    = nn.Linear(mu_dim, o_dim)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x, mu):
        '''
        x.size()  => (sample_size, x_dim)
        mu.size() => (sample_size, mu_dim)
        '''

        r = F.relu(self.mu2r(mu))
        phi = F.relu(self.xr2phi(torch.cat((x, r), 1)))
        mu = self.muphi2mu(mu, phi)
        o = F.relu(self.mu2o(mu))
        o = self.log_softmax(o)
        return o, mu
    
    def muphi2mu(self, mu, phi):
        '''
        すべての\alphaについて、\mu_t^{(\alpha)} = \alpha \mu_{t-1}^{(\alpha)} + (1-\alpha) \phi_t を同時に行う
            A_mask:   Kronecker product of (A, ones(1, phi_dim)),   shape => (1, mu_dim)
            phi_tile: Kronecker product of (ones(1, n_alpha), phi), shape => (sample_size, mu_dim)
        '''
        if self.gpu:
            A_mask = kronecker_product(self.A, torch.ones(1, self.phi_dim).cuda())
            phi_tile = kronecker_product(Variable(torch.ones(1, self.n_alpha).cuda()), phi)
        else:
            A_mask = kronecker_product(self.A, torch.ones(1, self.phi_dim))
            phi_tile = kronecker_product(Variable(torch.ones(1, self.n_alpha)), phi)

        # 要素積をとるためにA_maskをVariableに変換するが、A_maskは定数項なのでrequires_grad=Falseをつける
        A_mask = Variable(A_mask, requires_grad=False)
        mu = torch.mul(A_mask, mu) + torch.mul((1-A_mask), phi_tile)
        return mu


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


