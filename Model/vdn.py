import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VDN(nn.Module):
    def __init__(self, agents: dict, gamma, target_update, neighbors: dict = None):
        """

        :param agents: {智能体ID:DQN Agent对象}
        :param config: 训练配置
        """
        super().__init__()
        self.agents = agents
        self.gamma = gamma
        self.target_update = target_update
        self.learn_step = 0
        self.neighbors = neighbors if neighbors is not None else {}

        for ts_id, agent in self.agents.items():
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            agent.target_net.to(device)
            agent.policy_net.to(device)

    def update(self, replay_buffer, batch_size):
        """
        采样一批经验，集中更新
        :param replay_buffer:
        :param batch_size:
        :return:
        """
        if len(replay_buffer) < batch_size:
            return

        sample = replay_buffer.sample(batch_size)
        if len(sample) == 7:
            states, actions, next_states, rewards, dones, _, _ = sample
        else:
            states, actions, next_states, rewards, dones = sample
        # states, actions, next_states, rewards, dones, comms, next_comms = replay_buffer.sample(batch_size)
        for ts in states:
            states[ts] = states[ts].to(device)
            next_states[ts] = next_states[ts].to(device)
            actions[ts] = actions[ts].to(device)
            rewards[ts] = rewards[ts].to(device)
            dones[ts] = dones[ts].to(device)

        msgs = {}
        for ts_id, agent in self.agents.items():
            msgs[ts_id] = agent.encode_obs_tensor(states[ts_id])

        comms = {}
        for ts_id, agent in self.agents.items():
            if agent.comm_flag:
                neighbor_ids = self.neighbors.get(ts_id, [])
                neighbor_msgs = [msgs[nbr] for nbr in neighbor_ids if nbr in msgs]
                comm_vec = agent.aggregate_neighbors_tensor(msgs[ts_id], neighbor_msgs)  # 可微分
                comms[ts_id] = comm_vec
            else:
                comms[ts_id] = None

        # 计算联合Q值
        q_sum = 0
        for ts_id, agent in self.agents.items():
            if agent.comm_flag:
                combined = torch.cat([states[ts_id], comms[ts_id]], dim=-1)
            else:
                combined = states[ts_id]
            q_values = agent.policy_net(combined)
            q_selected = q_values.gather(1, actions[ts_id])
            q_sum += q_selected

        next_msgs = {}
        for ts_id, agent in self.agents.items():
            next_msgs[ts_id] = agent.encode_obs_tensor(next_states[ts_id]).detach()

        next_comms = {}
        for ts_id, agent in self.agents.items():
            if agent.comm_flag:
                neighbor_ids = self.neighbors.get(ts_id, [])
                neighbor_msgs = [next_msgs[nbr] for nbr in neighbor_ids if nbr in next_msgs]
                next_comm = agent.aggregate_neighbors_tensor(next_msgs[ts_id], neighbor_msgs).detach()
                next_comms[ts_id] = next_comm
            else:
                next_comms[ts_id] = None

        next_q_sum = 0
        for ts_id, agent in self.agents.items():
            if agent.comm_flag:
                combined_next = torch.cat([next_states[ts_id], next_comms[ts_id]], dim=-1)
            else:
                combined_next = next_states[ts_id]

            if agent.double:
                next_q_vals_policy = agent.policy_net(combined_next)
                next_actions = next_q_vals_policy.argmax(1, keepdim=True)
                next_q = agent.target_net(combined_next).gather(1, next_actions)
            else:
                next_q_vals = agent.target_net(combined_next)
                next_q = next_q_vals.max(1, keepdim=True)[0]
            next_q_sum += next_q
        # # 计算联合Q值
        # q_sum = 0
        # for ts_id in self.agents:
        #     if use_comm:
        #         combined = torch.cat([states[ts_id], comms[ts_id]], dim=-1).to(device)
        #     else:
        #         combined = states[ts_id].to(device)
        #     q = self.agents[ts_id].policy_net(combined)
        #     q_selected = q.gather(1, actions[ts_id].to(device))
        #     q_sum += q_selected
        #
        # next_q_sum = 0
        # for ts_id in self.agents:
        #     if use_comm:
        #         combined_next = torch.cat([next_states[ts_id], next_comms[ts_id]], dim=-1).to(device)
        #     else:
        #         combined_next = next_states[ts_id].to(device)
        #     if self.agents[ts_id].double:
        #         next_action = self.agents[ts_id].policy_net(combined_next).argmax(1, keepdim=True)
        #         next_q_max = self.agents[ts_id].target_net(combined_next).gather(1, next_action)
        #     else:
        #         next_q = self.agents[ts_id].target_net(combined_next)
        #         next_q_max = next_q.max(1, keepdim=True)[0]
        #     next_q_sum += next_q_max
        #
        global_reward = sum(rewards[ts_id].to(device) for ts_id in self.agents)
        # global_reward = sum(rewards[ts_id].to(device) for ts_id in self.agents) / len(self.agents)
        global_done = dones[next(iter(self.agents))].to(device)
        target = global_reward + self.gamma * next_q_sum * (1.0 - global_done)
        loss = nn.MSELoss()(q_sum, target)

        for agent in self.agents.values():
            agent.optimizer.zero_grad()
        loss.backward()
        for agent in self.agents.values():
            torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 10)
            agent.optimizer.step()
        self.learn_step += 1

        if self.learn_step % self.target_update == 0:
            for ts_id in self.agents:
                self.agents[ts_id].target_net.load_state_dict(self.agents[ts_id].policy_net.state_dict())
                self.agents[ts_id].save_model(self.learn_step)
