import torch
from torch import nn
from torch.nn.parameter import Parameter
import math

encoder_layers = 12
encoder_out_dim = 16
N = 4
T = 5
C = 24

layers = [3,6,9] # [1,...,12] 获取指定层的输出
encoder_output_list = []

D_1 = encoder_out_dim * len(layers)
D_2 = 3

print("T:",T, "\tN:",N, "\tC:",C)

print("encoder_out_dim:",encoder_out_dim, "\tlayers:",len(layers),"\tD_1:",D_1, "\tD_2:",D_2)

for i in layers:
    # conformer encoder
    encoder_output_list.append(torch.randn(T, encoder_out_dim))

# 声学信息
X_a = torch.cat(encoder_output_list, dim=-1)
print("X_a:",X_a.shape)


# 文本信息
# 为什么文本的T（seq_len）和音频的一样？ D_2是什么？
# 肯定不应该是[0,1,1,2,0]
# [[0,1,0,0,0],
# [0,0,0,2,0]]
#  (T, D_2=2)
# 应该是一整段的音频对应一整段的文本，所以T是一样的，D_2是文本的维度也就是词向量的维度？
X_t = torch.randn(T, D_2)
print("X_t:",X_t.shape)

W_a = []
V_a = []
W_t = []
V_t = []
S = []

for i in range(N):
    w1 = Parameter(torch.randn(D_1, C), requires_grad=True)
    W_a.append(w1)
    a = X_a.mm(w1)
    V_a.append(a)

    w2 = Parameter(torch.randn(D_2, C), requires_grad=True)
    W_t.append(w2)
    t = X_t.mm(w2)
    V_t.append(t)
    d_k = C/N

    s = torch.zeros(T)
    for j in range(T):
        s[j] = torch.dot(a[j], t[j]) / math.sqrt(d_k)
    S.append(s.unsqueeze(1))

W_a = torch.stack(W_a, dim=-1)
V_a = torch.stack(V_a, dim=-1)
W_t = torch.stack(W_t, dim=-1)
V_t = torch.stack(V_t, dim=-1)

print("W_a:",W_a.shape)
print("V_a:",V_a.shape)
print("W_t:",W_t.shape)
print("V_t:",V_t.shape)

S = torch.cat(S, dim=-1)

print("S:",S.shape)

W_td = Parameter(torch.randn(D_2, C-N), requires_grad=True)
# L_td = nn.Linear(D_2, C-N, bias=False)
V_td = torch.matmul(X_t, W_td)

print("W_td:",W_td.shape)
print("V_td:",V_td.shape)

Y_bm = torch.cat([S, V_td], dim=-1)

print("Y_bm:",Y_bm.shape)

