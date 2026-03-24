"""实现集中训练"""
import random
import torch
import numpy as np


class CentralizedReplayBuffer:
    def __init__(self, capacity: int, ts_ids: list):
        self.capacity = capacity
        self.ts_ids = ts_ids
        self.buffer = []
        self.position = 0

    def add(self, state: int, action: dict, next_state: dict, reward: dict, done: dict):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = {ts: [] for ts in self.ts_ids}
        actions = {ts: [] for ts in self.ts_ids}
        next_states = {ts: [] for ts in self.ts_ids}
        rewards = {ts: [] for ts in self.ts_ids}
        dones = {ts: [] for ts in self.ts_ids}

        for exp in batch:
            s, a, ns, r, d = exp
            for ts in self.ts_ids:
                states[ts].append(s[ts])
                actions[ts].append(a[ts])
                next_states[ts].append(ns[ts])
                rewards[ts].append(r[ts])
                dones[ts].append(d[ts])

        for ts in self.ts_ids:
            states[ts] = torch.tensor(np.array(states[ts]), dtype=torch.float32)
            actions[ts] = torch.tensor(np.array(actions[ts]), dtype=torch.long).unsqueeze(-1)
            next_states[ts] = torch.tensor(np.array(next_states[ts]), dtype=torch.float32)
            rewards[ts] = torch.tensor(np.array(rewards[ts]), dtype=torch.float32).unsqueeze(-1)
            dones[ts] = torch.tensor(np.array(dones[ts]), dtype=torch.float32).unsqueeze(-1)

        return states, actions, next_states, rewards, dones

    def __len__(self):
        return len(self.buffer)
