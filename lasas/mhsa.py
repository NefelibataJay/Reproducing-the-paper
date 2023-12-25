import math

import torch
from torch import nn

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, n_feat):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)

    def forward_qkv(self, query, key, value):
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, T, d_k)
        k = k.transpose(1, 2)  # (batch, head, T, d_k)
        v = v.transpose(1, 2)  # (batch, head, T, d_k)
        return q, k, v

    def forward_attention(self, value, scores):
        n_batch = value.size(0)
        attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(attn, value)  # (batch, head, T, d_k)
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k))  # (batch, T, d_model)
        return self.linear_out(x)  # (batch, T, d_model)

    def forward(self, query, key, value):
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores)

if __name__ == "__main__":
    n_head = 4
    n_feat = 16
    batch_size = 1
    T = 5
    mhsa = MultiHeadedAttention(n_head, n_feat)
    query = torch.randn(batch_size, T, n_feat)
    key = torch.randn(batch_size, T, n_feat)
    value = torch.randn(batch_size, T, n_feat)

    out = mhsa(query, key, value) # (batch, T, n_feat)
    