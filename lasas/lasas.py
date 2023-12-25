import math

import torch
from torch import nn

class LASAS(nn.Module):
    def __init__(self, a_dim, t_dim, out_dim , n_head):
        super(LASAS, self).__init__()
        assert out_dim % n_head == 0

        self.n_head = n_head
        self.d_k = out_dim // n_head
        
        self.acoustic_linear_list = torch.nn.ModuleList([nn.Linear(a_dim, out_dim) for i in range(n_head)])
        self.text_linear_list = torch.nn.ModuleList([nn.Linear(t_dim, out_dim) for i in range(n_head)])

        self.linear_td = nn.Linear(t_dim, out_dim-n_head)

    def forward(self, acoustic_emb, text_emb):
        """
        acoustic_emb: (batch, T, a_dim)
        text_emb: (batch, T, t_dim)
        """

        S = []
        for i in range(self.n_head):
            v_a = self.acoustic_linear_list[i](acoustic_emb)
            v_t = self.text_linear_list[i](text_emb)
            s = torch.matmul(v_a, v_t.transpose(1,2)) / math.sqrt(self.d_k)

            s = s.diagonal(dim1=-2, dim2=-1) # 取每个batch的对角线

            S.append(s.unsqueeze(-1))

        S = torch.cat(S, dim=-1)

        v_td = self.linear_td(text_emb)
        y_bm = torch.cat([S, v_td], dim=-1)

        return y_bm
        

if __name__ == "__main__":
    n_head = 4
    n_feat = 128
    a_dim = 32
    t_dim = 8
    batch_size = 2
    T = 9
    mhsa = LASAS(a_dim=a_dim, t_dim=t_dim, out_dim=n_feat, n_head=n_head)
    query = torch.randn(batch_size, T, a_dim)
    key = torch.randn(batch_size, T, t_dim)

    out = mhsa(query, key) # (batch, T, n_feat)
    