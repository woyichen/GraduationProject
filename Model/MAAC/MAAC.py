import torch
import torch.nn as nn


class MAAC(nn.Module):
    def __init__(self, obs_dim, action_dim, n_agents, hidden_dim):
        super().__init__()

        self.n_agents = n_agents
        self.input_dim = obs_dim + action_dim

        self.embed = nn.Linear(self.input_dim, hidden_dim)

        self.attention = MultiHeadAttention(hidden_dim, hidden_dim)

        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs_all, act_all):
        pass
