"""实现集中训练"""
import random
import torch
import numpy as np


class CentralizedReplayBuffer:
    def __init__(self, capacity: int,
                 ts_ids: list,
                 state_dim: int = None,
                 comm_dim: int = None,
                 neighbors: dict = None):
        self.capacity = capacity
        self.ts_ids = ts_ids
        self.state_dim = state_dim
        self.comm_dim = comm_dim
        self.store_comm = (comm_dim is not None)
        self.neighbors = neighbors
        self.buffer = []
        self.position = 0

    def add(self,
            state: int,
            action: dict,
            next_state: dict,
            reward: dict,
            done: dict,
            comm: dict = None,
            next_comm: dict = None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        if self.store_comm:
            if comm is None:
                comm = {}
            if next_comm is None:
                next_comm = {}
            full_comm = {ts: comm.get(ts, np.zeros(self.comm_dim)) for ts in self.ts_ids}
            full_next_comm = {ts: next_comm.get(ts, np.zeros(self.comm_dim)) for ts in self.ts_ids}
            self.buffer[self.position] = (state, action, next_state, reward, done, full_comm, full_next_comm)
            # self.buffer[self.position] = (state, action, next_state, reward, done, comm, next_comm)
        else:
            self.buffer[self.position] = (state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = {ts: [] for ts in self.ts_ids}
        actions = {ts: [] for ts in self.ts_ids}
        next_states = {ts: [] for ts in self.ts_ids}
        rewards = {ts: [] for ts in self.ts_ids}
        dones = {ts: [] for ts in self.ts_ids}

        if self.store_comm:
            comms = {ts: [] for ts in self.ts_ids}
            next_comms = {ts: [] for ts in self.ts_ids}
            for exp in batch:
                s, a, ns, r, d, c, nc = exp
                for ts in self.ts_ids:
                    states[ts].append(s[ts])
                    actions[ts].append(a[ts])
                    next_states[ts].append(ns[ts])
                    rewards[ts].append(r[ts])
                    dones[ts].append(d[ts])
                    comms[ts].append(c[ts])
                    next_comms[ts].append(nc[ts])

            for ts in self.ts_ids:
                states[ts] = torch.tensor(np.array(states[ts]), dtype=torch.float32)
                actions[ts] = torch.tensor(np.array(actions[ts]), dtype=torch.long).unsqueeze(-1)
                next_states[ts] = torch.tensor(np.array(next_states[ts]), dtype=torch.float32)
                rewards[ts] = torch.tensor(np.array(rewards[ts]), dtype=torch.float32).unsqueeze(-1)
                dones[ts] = torch.tensor(np.array(dones[ts]), dtype=torch.float32).unsqueeze(-1)
                comms[ts] = torch.tensor(np.array(comms[ts]), dtype=torch.float32)
                next_comms[ts] = torch.tensor(np.array(next_comms[ts]), dtype=torch.float32)
            return states, actions, next_states, rewards, dones, comms, next_comms
        else:
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
