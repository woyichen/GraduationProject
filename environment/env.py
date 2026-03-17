import gymnasium as gym
from gymnasium import spaces
import os
import sys
import random
import numpy as np

from environment.traffic_signal import TrafficSignal

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
import sumolib


class SumoEnv(gym.Env):
    def __init__(self,
                 # sumo,
                 net_file: str,
                 route_file: str,
                 skip_range: int,
                 simulation_time: float,
                 yellow_time: int = 3,
                 min_green_time: int = 10,
                 max_green_time: int = 60,
                 delta_rs_update_time: int = 1,
                 delta_time: int = 5,
                 use_neighbor: bool = False,
                 use_gui: bool = False,
                 reward_type="delta_waiting"
                 ):
        """

        :param net_file: 路网文件路径
        :param route_file: 路由文件路径
        :param skip_range: 随机跳过的时间范围
        :param simulation_time: 仿真时间
        :param yellow_time: 黄灯时间
        :param min_green_time: 最小绿灯时间
        :param max_green_time: 最大绿灯时间
        :param delta_rs_update_time: 相位更新间隔
        :param delta_time:仿真步数
        :param use_neighbor: 是否有通讯
        :param use_gui: 是否使用GUI界面
        :param reward_type: 奖励函数类型
        """

        super(SumoEnv, self).__init__()
        # self.sumo = sumo
        self.sumo = None
        # 路网文件
        self.net_file = net_file
        # 路由文件
        self.route_file = route_file
        # 随机跳过时间范围
        self.skip_range = skip_range
        # 总模拟时间
        self.simulation_time = simulation_time
        # 是否使用GUI
        self.use_gui = use_gui
        self.sumoBinary = 'sumo-gui' if use_gui else 'sumo'
        # 初始化训练状态
        self.train_state = None
        # 初始化上一个相位状态
        self.last_phase_state = None
        # 初始化相位切换时间
        self.change_action_time = None
        self.delta_time = delta_time
        traci.start([sumolib.checkBinary('sumo'), '-n', self.net_file])
        self.ts_ids = traci.trafficlight.getIDList()
        self.reward_type = reward_type
        traci.close()
        self.traffic_signals = {
            ts_id: TrafficSignal(
                sumo=self.sumo,
                net_file=net_file,
                ts_id=ts_id,
                yellow_time=yellow_time,
                min_green=min_green_time,
                max_green=max_green_time,
                delta_rs_update_time=delta_rs_update_time,
                use_neighbor=use_neighbor,
                reward_type=self.reward_type
            )
            for ts_id in self.ts_ids  # 对每个路口创建一个 TrafficSignal 对象
        }

        # 动作空间
        self.action_space = spaces.Dict({})
        # self.action_space = spaces.Dict({
        #     ts_id: spaces.Discrete(len(self.traffic_signals[ts_id].all_green_phases))
        #     for ts_id in self.ts_ids
        # })
        # 状态空间
        self.observation_space = spaces.Dict({})
        # self.observation_space = spaces.Dict({
        #     ts_id: ts.observation_space
        #     for ts_id, ts in self.traffic_signals.items()
        # })
        # self.observation_space = spaces.Box(
        #     low=0, high=1, shape=(len(self.ts_ids),), dtype=np.float32
        # )

    def _random_skip(self, skip_range):
        for ts in self.traffic_signals.values():
            rand_idx = random.randint(0, len(ts.all_green_phases) - 1)
            ts.green_phase = ts.all_green_phases[rand_idx]
            ts.yellow_phase = None
            self.sumo.trafficlight.setRedYellowGreenState(ts.ts_id, ts.green_phase.state)
            ts.phase_time = 0
        skip_seconds = random.randint(0, skip_range)
        for _ in range(skip_seconds):
            action = {
                ts_id: random.randint(0, len(ts.all_green_phases) - 1)
                for ts_id, ts in self.traffic_signals.items()
            }
            self.step(action)
        return self._get_states()

    def reset(self, seed=None, options=None):
        """
        重置环境
        :return:
        """
        sumo_cmd = [sumolib.checkBinary(self.sumoBinary),
                    '-n', self.net_file,
                    '-r', self.route_file,
                    '--time-to-teleport', '1000']
        if self.use_gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])
        traci.start(sumo_cmd)
        self.sumo = traci

        for ts in self.traffic_signals.values():
            ts.sumo = self.sumo
            ts.initialize()
            # ts.green_phase = None
            # ts.yellow_phase = None
            # ts.phase_time = 0
            # ts.last_waiting = 0
            # ts.next_action_time = 0
        # 动作空间
        self.action_space = spaces.Dict({
            ts_id: spaces.Discrete(len(self.traffic_signals[ts_id].all_green_phases))
            for ts_id in self.ts_ids
        })
        # 状态空间
        self.observation_space = spaces.Dict({
            ts_id: ts.observation_space
            for ts_id, ts in self.traffic_signals.items()
        })
        state = self._random_skip(self.skip_range)

        return state, {}

    def _get_states(self):
        """
        获取所有交通信号灯的状态
        :return:
        """
        states = {}
        for ts_id, ts in self.traffic_signals.items():
            states[ts_id] = ts.compute_state()
        return states

    def step(self, actions):
        rewards = {}
        states = {}
        # done = False

        # 执行每个路口动作
        for ts_id, action in actions.items():
            self.traffic_signals[ts_id].apply_action(action)
        # # 执行SUMO仿真一步
        # self.sumo.simulationStep()
        # 执行多步SUMO step
        for _ in range(self.delta_time):
            self.sumo.simulation.step()
            for ts in self.traffic_signals.values():
                ts.update_phase_time()
        for ts_id, ts in self.traffic_signals.items():
            states[ts_id] = ts.compute_state()
            rewards[ts_id] = ts.compute_reward()
        done = self._compute_done()
        return states, rewards, done, False, {}

    def render(self):
        pass

    def _compute_done(self):
        """
        检查是否结束
        :return:
        """
        current_time = self.sumo.simulation.getTime()
        if current_time > self.simulation_time:
            done = True
        else:
            done = False
        return done

    def close(self):
        if self.sumo is not None:
            self.sumo.close()


if __name__ == "__main__":
    net_file = '../nets/osm.net.xml.gz'
    route_file = '../nets/osm.passenger.rou.xml'
    SumoEnv(
        net_file=net_file,
        route_file=route_file,
        skip_range=10,
        simulation_time=1000,
        yellow_time=3,
        min_green_time=10,
        max_green_time=60,
        delta_rs_update_time=1,
        use_neighbor=False,
        use_gui=False
    )
