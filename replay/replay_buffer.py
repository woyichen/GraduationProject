import random
import torch
import numpy as np


class ReplayBuffer:
    def __init__(self,
                 capacity: int,
                 mode: str = 'single',
                 ts_ids=None,
                 state_dim=None,
                 action_dim=None):
        """

        :param capacity: 最大容量
        :param mode: ['single','independent','centralized']
        :param ts_ids: 多智能体ID列表
        :param state_dim: 状态维度
        """
        self.capacity = capacity
        self.mode = mode
        self.position = 0
        self.size = 0
        self.action_dim = action_dim

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if mode in ['single', 'independent']:
            self.storage = [None] * capacity

        elif mode in ['centralized']:
            assert ts_ids is not None
            assert state_dim is not None
            self.ts_ids = ts_ids
            self.n_agents = len(ts_ids)

            self.states = torch.zeros(capacity, self.n_agents, state_dim)
            self.next_states = torch.zeros(capacity, self.n_agents, state_dim)
            self.actions = torch.zeros(capacity, self.n_agents, 1)
            self.rewards = torch.zeros(capacity, self.n_agents, 1)
            self.dones = torch.zeros(capacity, self.n_agents, 1)
            self.masks = torch.zeros(capacity, self.n_agents, 1)

        else:
            raise ValueError("Invalid mode")

    def add(self, state, action, next_state, reward, done, action_mask=None):
        if self.mode in ['single', 'independent']:
            if action_mask is None:
                mask = torch.ones(self.action_dim)
            else:
                mask = torch.tensor(action_mask)
            self.storage[self.position] = (
                torch.tensor(state, dtype=torch.float32),
                torch.tensor([action], dtype=torch.long),
                torch.tensor(next_state, dtype=torch.float32),
                torch.tensor([reward], dtype=torch.float32),
                torch.tensor([done], dtype=torch.float32),
                mask
            )

        elif self.mode in ['centralized']:
            for i, ts_id in enumerate(self.ts_ids):
                self.states[self.position, i] = torch.tensor(state[ts_id])
                self.next_states[self.position, i] = torch.tensor(next_state[ts_id])
                self.actions[self.position, i] = action[ts_id]
                self.rewards[self.position, i] = reward[ts_id]
                self.dones[self.position, i] = done[ts_id]

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size)
        if self.mode in ['single', 'independent']:
            batch = [self.storage[i] for i in indices]
            state, action, next_state, reward, done, mask = zip(*batch)

            return (
                torch.stack(state).to(self.device),
                torch.stack(action).to(self.device),
                torch.stack(next_state).to(self.device),
                torch.stack(reward).to(self.device),
                torch.stack(done).to(self.device),
                torch.stack(mask).to(self.device),
            )
        elif self.mode in ['centralized']:
            return (
                self.states[indices].to(self.device),
                self.actions[indices].to(self.device),
                self.next_states[indices].to(self.device),
                self.rewards[indices].to(self.device),
                self.dones[indices].to(self.device),
            )

    def __len__(self):
        return self.size
