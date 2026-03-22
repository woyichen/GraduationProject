import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    MAAC核心通讯模块
    """

    def __init__(self, input_dim, output_dim, embed_dim, n_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads

        self.q = nn.Linear(input_dim, embed_dim)
        self.k = nn.Linear(input_dim, embed_dim)
        self.v = nn.Linear(input_dim, embed_dim)

        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        B, N, _ = x.shape
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        Q = Q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        return self.fc_out(out)
