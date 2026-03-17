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
                 network_file: str = ''):
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

        self.network_file = network_file
        # policy网络
        self.policy_net = Network(self.state_dim, self.n_action, hidden_dim).to(device)
        # target网络
        self.target_net = Network(self.state_dim, self.n_action, hidden_dim).to(device)
        # 加载权重
        if network_file:
            self.policy_net.load_state_dict(torch.load(network_file, map_location=device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # 训练部署
        self.learn_steps = 0

        # self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)

    def select_action(self, state, steps_done, invalid_action):
        original_state = state.copy()

        state = torch.from_numpy(state)
        if self.mode == 'train':
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * steps_done / self.eps_decay)
            if sample > eps_threshold:
                with torch.no_grad():
                    _, sorted_indices = torch.sort(self.policy_net(state), descending=True)
                    if invalid_action:
                        return sorted_indices[1]
                    else:
                        return sorted_indices[0]
            else:
                decrease_state = [(original_state[0] + original_state[4]) / 2,
                                  (original_state[1] + original_state[5]) / 2,
                                  (original_state[2] + original_state[6]) / 2,
                                  (original_state[3] + original_state[7]) / 2]
                congest_phase = [i for i, s in enumerate(decrease_state) if abs(s - 1) < 1e-2]
                if len(congest_phase) > 0 and invalid_action is False:
                    return random.choice(congest_phase)
                else:
                    return random.randrange(self.n_action)
        else:
            with torch.no_grad():
                _, sorted_indices = torch.sort(self.policy_net(state), descending=True)
                if invalid_action:
                    return sorted_indices[1]
                else:
                    return sorted_indices[0]

    def learn(self):
        if self.mode == "train":
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=0.00025)
            if self.replay.steps_done <= 10000:
                return
            transitions = self.replay.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action).view(self.batch_size, 1)
            next_state_batch = torch.cat(batch.next_state)
            reward_batch = torch.cat(batch.reward).view(self.batch_size, 1)
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)
            with torch.no_grad():
                argmax_action = self.policy_net(next_state_batch).max(1)[1].view(self.batch_size, 1)
                expected_state_action_values = reward_batch + self.gamma * \
                                               self.target_net(next_state_batch).gather(1, argmax_action)
            loss = loss_fn(state_action_values, expected_state_action_values)
            optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
            self.learn_steps += 1

            if self.learn_steps % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                time = str(datetime.now()).split(' ')[0]
                time = time.replace('-', '').replace(' ', '_').replace(':', '')
                torch.save(self.policy_net.state_dict(), 'weights/weights_{0}{1}.pth'.format(time, self.learn_steps))
