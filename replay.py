from collections import namedtuple, deque
import numpy as np
import torch

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReplayBuffer:
    def __init__(self,
                 capacity: int,
                 mode: str = "single",
                 ts_ids=None,
                 state_dim=None,
                 action_dim=None
                 ):
        """

        :param capacity: 最大容量
        :param mode: ['single','independent','centralized']
        :param ts_ids: 多智能体ID列表
        :param state_dim: 状态维度
        :param action_dim: 动作维度
        """
        self.capacity = capacity
        self.mode = mode
        self.position = 0
        self.size = 0
        if mode in ['single', 'independent']:
            self.storage = [None] * capacity

        elif mode == 'centralized':
            assert ts_ids is not None
            assert state_dim is not None

            self.ts_ids = ts_ids
            self.n_agents = len(ts_ids)

            self.states = torch.zeros(capacity, self.n_agents, state_dim)
            self.next_states = torch.zeros(capacity, self.n_agents, state_dim)
            self.actions = torch.zeros(capacity, self.n_agents, 1)
            self.rewards = torch.zeros(capacity, self.n_agents, 1)
        else:
            raise ValueError("Mode must be in ['single','independent','centralized']")

    def add(self, state, action, next_state, reward):
        if self.mode in ["single", "independent"]:
            state = torch.from_numpy(state).float().unsqueeze(0)
            next_state = torch.from_numpy(next_state).float().unsqueeze(0)
            action = torch.tensor([[action]])
            reward = torch.tensor([[reward]], dtype=torch.float32)

            self.storage[self.position] = (state, action, next_state, reward)
        elif self.mode in ["centralized"]:
            for i, ts_id in enumerate(self.ts_ids):
                self.states[self.position, i] = torch.from_numpy(state[ts_id])
                self.next_states[self.position, i] = torch.from_numpy(next_state[ts_id])
                self.actions[self.position, i] = action[ts_id]
                self.rewards[self.position, i] = reward[ts_id]
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size)
        if self.mode in ["single", "independent"]:
            batch = [self.storage[i] for i in indices]
            state, action, next_state, reward = zip(*batch)

            return (
                torch.cat(state),
                torch.cat(action),
                torch.cat(next_state),
                torch.cat(reward)
            )
        elif self.mode in ["centralized"]:
            return (
                self.states[indices],
                self.actions[indices],
                self.next_states[indices],
                self.rewards[indices]
            )

    def __len__(self):
        return self.size


if __name__ == "__main__":
    pass
