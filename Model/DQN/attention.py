import torch
import torch.nn as nn


class CommAttention(nn.Module):
    """
    核心通讯模块
    """

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.q = nn.Linear(input_dim, embed_dim)
        self.k = nn.Linear(input_dim, embed_dim)
        self.v = nn.Linear(input_dim, embed_dim)
        self.out = nn.Linear(embed_dim, input_dim)

    def forward(self, x, mask=None):
        # print(x.shape)
        B, N, _ = x.shape
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = self.out(out)

        return out[:, 0, :]
