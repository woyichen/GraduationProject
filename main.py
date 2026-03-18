import os
import sys
from absl import app
from absl import flags

FLAGS = flags.FLAGS
# 仿真开始随机跳过的时间范围
flags.DEFINE_integer('skip_range', 10, 'time range for skip randomly at the beginning')
# 每次episode的总仿真时间
flags.DEFINE_float('simulation_time', 3000, 'time for simulation')
# 黄灯持续时间
flags.DEFINE_integer('yellow_time', 2, 'time for yellow phase')
# 最小绿灯时间
flags.DEFINE_integer('min_green_time', 10, 'time for min green phase')
# 最长绿灯时间
flags.DEFINE_integer('max_green_time', 120, 'time for max green phase')
# 计算奖励的时间间隔，动作的奖励无法立即体现
flags.DEFINE_integer('delta_rs_update_time', 5, 'time for calculate reward')
# 模拟步数
flags.DEFINE_integer('delta_time', 5, '')
# 路网文件
flags.DEFINE_string('net_file', './nets/osm.net.xml.gz', 'net file')
# 车辆路由文件
flags.DEFINE_string('route_file', './nets/osm.passenger.rou.xml', 'route file')
# 是否有通讯
flags.DEFINE_bool('use_neighbor', False, '')
# 是否使用GUI
flags.DEFINE_bool('use_gui', True, 'use sumo-gui instead of sumo')
# 奖励函数类型
flags.DEFINE_string('reward_type', "delta_waiting", '')
flags.DEFINE_string('mode', 'train', 'train or test')
# 训练中的episodes总数
flags.DEFINE_integer('num_episodes', 301, 'number of episodes')

# ε-贪心策略的初始探索率
flags.DEFINE_float('eps_start', 1.0, '')
# 最小探索率
flags.DEFINE_float('eps_end', 0.1, '')
# ε 衰减的步数（指数衰减公式中的分母）
flags.DEFINE_integer('eps_decay', 200000, '')
# 目标网路更新频率
flags.DEFINE_integer('target_update', 1000, '')
# 折扣因子 γ
flags.DEFINE_float('gamma', 0.95, '')
# 经验回放中采样的批次大小
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('buffer_size', 20000, '')
# 权重存储位置
flags.DEFINE_string('network_file', './weights/weights.pth', 'net file')

flags.DEFINE_list('tls_id', ['cluster_366489708_9203769172',
                             '2187544218',
                             '2187544217',
                             'cluster_2178819402_2189318888',
                             '2187544212',
                             '2187544213',
                             'cluster_2178819374_4839352770_4839352772',
                             'cluster_2187544206_4839352776',
                             'cluster_2187544208_4839352781'], '')

import torch
import torch.nn as nn
import traci
import sumolib

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import environment
from DQN.DQN import Agent
from replay import ReplayBuffer


def main(argv):
    del argv

    env = environment.SumoEnv(
        net_file=FLAGS.net_file,
        route_file=FLAGS.route_file,
        skip_range=FLAGS.skip_range,
        simulation_time=FLAGS.simulation_time,
        yellow_time=FLAGS.yellow_time,
        min_green_time=FLAGS.min_green_time,
        max_green_time=FLAGS.max_green_time,
        delta_rs_update_time=FLAGS.delta_rs_update_time,
        delta_time=FLAGS.delta_time,
        use_neighbor=FLAGS.use_neighbor,
        use_gui=FLAGS.use_gui,
        reward_type=FLAGS.reward_type,
    )
    states, _ = env.reset()

    agents = {}
    buffers = {}

    for ts_id, state in states.items():
        state_dim = len(state)
        action_dim = env.action_space[ts_id].n

        buffers[ts_id] = ReplayBuffer(FLAGS.buffer_size)

        agents[ts_id] = Agent(
            mode=FLAGS.mode,
            replay=buffers[ts_id],
            target_update=FLAGS.target_update,
            gamma=FLAGS.gamma,
            eps_start=FLAGS.eps_start,
            eps_end=FLAGS.eps_end,
            eps_decay=FLAGS.eps_decay,
            batch_size=FLAGS.batch_size,
            state_dim=state_dim,
            action_dim=action_dim,
        )
    env.close()

    for episode in range(FLAGS.num_episodes):
        states, _ = env.reset()
        done = False

        # n = 0
        while not done:
            # print(n)
            # n += 1
            actions = {}
            for ts_id in agents:
                # print()
                actions[ts_id] = agents[ts_id].select_action(
                    states[ts_id],
                    buffers[ts_id].size,
                    False
                )
            # print(actions)
            # for id in FLAGS.tls_id:
            #     print(actions[id], end='')
            # print()
            next_states, rewards, done, _, _ = env.step(actions)

            for ts_id in agents:
                buffers[ts_id].add(
                    states[ts_id],
                    actions[ts_id],
                    next_states[ts_id],
                    rewards[ts_id]
                )
                agents[ts_id].learn()

            states = next_states
        print(f"Episode: {episode}, Rewards: {rewards}")
        env.close()


if __name__ == "__main__":
    app.run(main)
