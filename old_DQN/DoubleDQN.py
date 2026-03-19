import math
import random
import torch
import torch.nn as nn
from datetime import datetime
from collections import namedtuple

from .networks import Network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class Agent:
    def __init__(self,
                 mode: str,
                 replay,
                 target_update: int,
                 gamma: float,
                 eps_start: float,  # ε的初始值
                 eps_end: float,  # ε的最小值
                 eps_decay: float,  # ε衰减的步数（指数衰减速率）
                 batch_size: int,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim=512,
                 network_file: str = None):
        self.mode = mode
        self.replay = replay
        self.target_update = target_update
        # 折扣因子
        self.gamma = gamma
        # ε greedy参数
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        # 状态维度
        self.state_dim = state_dim
        # 动作数量
        self.n_action = action_dim
        self.hidden_dim = hidden_dim

        # policy网络
        self.policy_net = Network(self.state_dim, self.n_action, hidden_dim).to(device)
        # target网络
        self.target_net = Network(self.state_dim, self.n_action, hidden_dim).to(device)
        # 加载权重
        if network_file:
            self.policy_net.load_state_dict(torch.load(network_file, map_location=device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        # 训练部署
        self.learn_steps = 0

    def select_action(self, state, steps_done, invalid_action):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        if self.mode == 'train':
            r = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * steps_done / self.eps_decay)
            print(f"{r:.2f}/{eps_threshold:.2f}", end=' ')
            if r < eps_threshold:
                action = random.randrange(self.n_action)
                # print(f"{action}/{self.n_action}", end=' ')
                return action
            else:
                with torch.no_grad():
                    q_values = self.policy_net(state)
                    action = q_values.argmax(dim=1).item()
                    if invalid_action:
                        sorted_idx = torch.argsort(q_values, descending=True)
                        return sorted_idx[0, 1].item()
                    return action
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.argmax(dim=1).item()

    def learn(self):
        if self.replay.size < self.batch_size:
            return
        state_batch, action_batch, next_state_batch, reward_batch = self.replay.sample(self.batch_size)
        state_batch = state_batch.to(device).float()
        next_state_batch = next_state_batch.to(device).float()
        action_batch = action_batch.to(device).long()
        reward_batch = reward_batch.to(device).float()
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_state_batch).gather(1, next_actions)
            target_q = reward_batch + self.gamma * next_q
        loss = self.loss_fn(state_action_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        self.learn_steps += 1

        if self.learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            time = str(datetime.now()).split(' ')[0]
            time = time.replace('-', '').replace(' ', '_').replace(':', '')
            torch.save(self.policy_net.state_dict(), f'weights/weights_{time}_{self.learn_steps}.pth')
