from collections import namedtuple, deque
import numpy as np
import torch

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReplayBuffer:
    def __init__(self,
                 capacity: int,
                 ts_ids
                 ):
        self.capacity = capacity
        self.ts_ids = ts_ids
        self.storage = []
        self.position = 0

    def add(self, state, action, next_state, reward):
        state_tensor = {
            k: torch.from_numpy(v).float()
            for k, v in state.items()
        }
        next_state_tensor = {
            k: torch.from_numpy(v).float()
            for k, v in next_state.items()
        }
        action_tensor = {
            k: torch.tensor(v)
            for k, v in action.items()
        }
        reward_tensor = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in reward.items()
        }
        data = (state_tensor, action_tensor, next_state_tensor, reward_tensor)

        if len(self.storage) < self.capacity:
            self.storage.append(data)
        else:
            self.storage[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.storage), batch_size)
        batch = [self.storage[i] for i in indices]
        return batch

    def __len__(self):
        return len(self.storage)
