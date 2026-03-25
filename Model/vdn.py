import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VDN(nn.Module):
    def __init__(self, agents: dict, gamma, target_update):
        """

        :param agents: {智能体ID:DQN Agent对象}
        :param config: 训练配置
        """
        super().__init__()
        self.agents = agents
        self.gamma = gamma
        self.target_update = target_update
        self.learn_step = 0

        for ts_id, agent in self.agents.items():
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            agent.target_net.to(device)
            agent.policy_net.to(device)
            # agent.optimizer = optim.Adam(agent.policy_net.parameters(), lr=agent.lr)

    def update(self, replay_buffer, batch_size):
        """
        采样一批经验，集中更新
        :param replay_buffer:
        :param batch_size:
        :return:
        """
        if len(replay_buffer) < batch_size:
            return

        states, actions, next_states, rewards, dones = replay_buffer.sample(batch_size)
        # 计算联合Q值
        q_sum = 0
        for ts_id in self.agents:
            q = self.agents[ts_id].policy_net(states[ts_id].to(device))
            q_selected = q.gather(1, actions[ts_id].to(device))
            q_sum += q_selected

        next_q_sum = 0
        for ts_id in self.agents:
            next_q = self.agents[ts_id].target_net(next_states[ts_id].to(device))
            next_q_max = next_q.max(1, keepdim=True)[0]
            next_q_sum += next_q_max

        global_reward = sum(rewards[ts_id].to(device) for ts_id in self.agents)
        global_done = dones[next(iter(self.agents))].to(device)
        target = global_reward + self.gamma * next_q_sum * (1.0 - global_done)
        loss = nn.MSELoss()(q_sum, target)

        for ts_id in self.agents:
            self.agents[ts_id].optimizer.zero_grad()
        loss.backward()
        for ts_id in self.agents:
            torch.nn.utils.clip_grad_norm_(self.agents[ts_id].policy_net.parameters(), 10)
            self.agents[ts_id].optimizer.step()
        self.learn_step += 1

        if self.learn_step % self.target_update == 0:
            for ts_id in self.agents:
                self.agents[ts_id].target_net.load_state_dict(self.agents[ts_id].policy_net.state_dict())
                self.agents[ts_id].save_model(self.learn_step)
