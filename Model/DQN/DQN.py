import os
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from Model.DQN.networks import Network
from Model.DQN.attention import CommAttention
from replay.replay_buffer import ReplayBuffer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Agent:
    def __init__(self,
                 ts_id,
                 state_dim,
                 action_dim,
                 hidden_dim,
                 lr,
                 gamma,
                 batch_size,
                 eps_start,
                 eps_end,
                 eps_decay,
                 target_update,
                 double,
                 save_path,
                 comm_flag=False,
                 n_agents=0,
                 comm_embed_dim=64,
                 neighbors=None,
                 ):
        self.ts_id = ts_id
        self.policy_net = Network(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = Network(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.lr = lr
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.learn_step = 0
        self.target_update = target_update
        self.action_dim = action_dim
        self.double = double

        self.save_path = save_path
        self.comm_flag = comm_flag
        if comm_flag:
            self.n_agents = n_agents
            self.comm_embed_dim = comm_embed_dim
            self.state_dim = state_dim
            self.obs_encoder = nn.Linear(state_dim, comm_embed_dim).to(device)
            self.comm = CommAttention(comm_embed_dim, comm_embed_dim).to(device)
            # self.obs_embed = nn.Linear(self.state_dim, self.comm_embed_dim).to(device)
            input_dim = self.state_dim + self.comm_embed_dim
            self.policy_net = Network(input_dim, action_dim, hidden_dim).to(device)
            self.target_net = Network(input_dim, action_dim, hidden_dim).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = optim.Adam(list(self.policy_net.parameters())
                                        + list(self.obs_encoder.parameters())
                                        + list(self.comm.parameters()), lr=self.lr)

    def select_action(self, state, step, comm_vec=None, action_mask=None):
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * step / self.eps_decay)

        if random.random() < eps:
            if action_mask is not None:
                valid = torch.where(torch.tensor(action_mask) == 1)[0]
                return valid[torch.randint(len(valid), (1,))].item()
            return random.randrange(self.action_dim)
        with torch.no_grad():
            if self.comm_flag and comm_vec is not None:
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                comm_t = torch.tensor(comm_vec, dtype=torch.float32).unsqueeze(0).to(device)
                combined = torch.cat([state_t, comm_t], dim=1)
                q = self.policy_net(combined).squeeze(0)
            else:
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q = self.policy_net(state_t).squeeze(0)
            if action_mask is not None:
                mask = torch.tensor(action_mask, device=q.device)
                q[mask == 0] = -1e9
            return q.argmax().item()

    def learn(self, replay: ReplayBuffer):
        if replay.size < self.batch_size:
            return

        s, a, s_, r, d, mask = replay.sample(self.batch_size)
        s = s.float().to(device)
        s_ = s_.float().to(device)
        a = a.long().to(device)
        r = r.float().to(device)
        d = d.float().to(device)

        # a = a.unsqueeze(1)
        q = self.policy_net(s).gather(1, a)

        with torch.no_grad():
            if self.double:
                next_q_values = self.policy_net(s_)
                next_q_values[mask == 0] = -1e9
                next_action = next_q_values.argmax(1, keepdim=True)
                next_q = self.target_net(s_).gather(1, next_action)
            else:
                next_q_values = self.target_net(s_)
                next_q_values[mask == 0] = 1e-9
                next_q = next_q_values.max(1, keepdim=True)[0]
            target = r + self.gamma * next_q * (1 - d)
        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        self.learn_step += 1

        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.save_model(self.learn_step)

    def save_model(self, step):
        import os
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, self.ts_id), exist_ok=True)
        save_dict = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': step
        }
        if self.comm_flag:
            save_dict['obs_encoder'] = self.obs_encoder.state_dict()
            save_dict['comm_net'] = self.comm.state_dict()
        torch.save(save_dict, f"{self.save_path}/{self.ts_id}/_{step}.pth")

    def load_model(self, file_path):
        checkpoint = torch.load(file_path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.learn_step = checkpoint.get('step', 0)
        if self.comm_flag and 'obs_encoder' in checkpoint:
            self.obs_encoder.load_state_dict(checkpoint['obs_encoder'])
            self.comm.load_state_dict(checkpoint['comm_net'])

    def encode_obs(self, obs):
        if not self.comm_flag:
            return obs
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        msg = self.obs_encoder(obs_t)
        return msg.squeeze(0).detach().cpu().numpy()

    def aggregate_neighbors(self, self_msg, neighbor_msgs):
        if not self.comm_flag or len(neighbor_msgs) == 0:
            return np.zeros(self.obs_encoder.out_features)
        msgs = [self_msg] + neighbor_msgs
        msgs_t = torch.tensor(np.stack(msgs), dtype=torch.float32).unsqueeze(0).to(device)
        agg = self.comm(msgs_t)
        return agg.squeeze(0).detach().cpu().numpy()
