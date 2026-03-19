import os
import sys
from typing import Callable, List, Union

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the old_environment variable 'SUMO_HOME'")
import numpy as np
from gymnasium import spaces
from queue import Queue


class TrafficSignal:
    MIN_GAP = 2.5

    def __init__(
            self,
            env,
            ts_id: str,
            delta_time: int,
            yellow_time: int,
            min_green: int,
            max_green: int,
            begin_time: int,
            reward_fn: Union[str, Callable, List],
            reward_weights: List[float],
            sumo,
            neighbor: bool = False
    ):
        """

        :param env: 交通信号的环境
        :param ts_id: 交通信号的ID
        :param delta_time: 动作之间的时间间隔
        :param yellow_time: 黄灯相位的持续时间
        :param min_green: 绿灯相位的最小持续时间
        :param max_green: 绿灯相位的最大持续时间
        :param begin_time: 交通信号开始运行的时间
        :param reward_fn: 奖励函数
        :param reward_weights: 奖励函数的权重
        :param sumo: Sumo实例
        """
        # 交通信号ID
        self.id = ts_id
        # 所属环境
        self.env = env
        # 动作间隔
        self.delta_time = delta_time
        # 黄灯时长
        self.yellow_time = yellow_time
        # 最小绿灯时间
        self.min_green = min_green
        # 最大绿灯时间
        self.max_green = max_green
        # 当前绿灯相位索引
        self.green_phase = 0
        # 当前是否是黄灯相位
        self.is_yellow = False
        # 自上次相位切换以来的时间
        self.time_since_last_phase_change = 0
        # 下次执行动作的时间
        self.next_action_time = begin_time
        # 上时刻的总等待时间
        self.last_ts_waiting_time = None
        # 奖励函数
        self.reward_fn = reward_fn
        # 奖励权重
        self.reward_weights = reward_weights
        # sumo链接实例
        self.sumo = sumo
        if type(self.reward_fn) is list:
            # 奖励维度
            self.reward_dim = len(self.reward_fn)
            self.reward_list = [self._get_reward_fn_from_string(reward_fn)
                                for reward_fn in self.reward_fn]
        else:
            self.reward_dim = 1
            self.reward_list = [self._get_reward_fn_from_string(self.reward_fn)]

        if self.reward_weights is not None:
            self.reward_dim = 1
        # 奖励空间
        self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,), dtype=np.float32)
        # 初始化观测函数
        self.observation_fn = self.env.observation_class(self)
        # 构建交通信号相位
        self._build_phases()
        # 入口车道，去重，保持顺序
        self.lanes = list(dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id)))
        # 出口车道
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        # 所有车道长度
        self.lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes}
        # 观测空间由观测函数决定
        self.observation_space = self.observation_fn.observation_space()
        # 动作空间
        self.action_space = spaces.Discrete(self.num_green_phases)
        # 是否使用邻居信息
        self.neighbor_flag = neighbor
        if self.neighbor_flag:
            # 邻接信息
            self.neighbor = None
            self._set_neighbors()

    def _set_neighbors(self):
        """
        获取邻接的路口
        :return:
        """
        neighbors = set()
        tls_ids = set(self.sumo.trafficlight.getIDList())
        # print(tls_ids)
        # print(len(tls_ids))
        # links = self.sumo.trafficlight.getControlledLinks(self.ts_id)
        # print(links)
        # for link in links:
        #     print(link)
        path = set()
        path.add(self.id)
        q = Queue()
        q.put(self.id)
        while not q.empty():
            out_edges = self.net.getNode(q.get()).getOutgoing()
            for edge in out_edges:
                node = edge.getToNode().getID()
                if node not in path:
                    if node in tls_ids:
                        neighbors.add(node)
                    else:
                        q.put(node)
                    path.add(node)

        self.neighbors = list(neighbors)

    def _get_reward_fn_from_string(self, reward_fn):
        if type(reward_fn) is str:
            if reward_fn in TrafficSignal.reward_fns.keys():
                return TrafficSignal.reward_fns[reward_fn]
            else:
                raise NotImplementedError(f"Reward function {reward_fn} not implemented")
        return reward_fn

    def _build_phases(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        if self.env.fixed_ts:
            self.num_green_phases = len(phases) // 2
            return
        # 绿灯相位
        self.green_phases = []

        self.yellow_dict = {}
        # 提取绿灯相位
        for phase in phases:
            state = phase.state
            if "y" not in state and (state.count("r") + state.count("s") != len(state)):
                self.green_phases.append(self.sumo.trafficlight.Phase(60, state))
        # 绿灯相位数量
        self.num_green_phases = len(self.green_phases)
        # 所有相位
        self.all_phases = self.green_phases.copy()
        # 生成绿灯相位对应的黄灯相位
        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j:
                    continue
                yellow_state = ""
                for s in range(len(p1.state)):
                    if (p1.state[s] == "G" or p1.state[s] == "g") and (p2.state[s] == 'r' or p2.state[s] == 's'):
                        yellow_state += "y"
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i, j)] = len(self.all_phases)
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    @property
    def time_to_act(self):
        """当前时间步是否应该执行动作"""
        return self.next_action_time == self.env.sim_step

    def update(self):
        """更新交通信号状态"""
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.time_since_last_phase_change:
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False

    def set_next_phase(self, new_phase: int):
        if new_phase == self.green_phase and self.time_since_last_phase_change >= self.max_green:
            new_phase = (self.green_phase + 1) % self.num_green_phases

        if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            yellow_phase_index = self.yellow_dict[(self.green_phase, new_phase)]
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[yellow_phase_index].state)
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0

    def compute_observation(self):
        """交通信号的观测值"""
        return self.observation_fn()

    def compute_reward(self) -> Union[float, np.ndarray]:
        if self.reward_dim == 1:
            self.last_reward = self.reward_list[0](self)
        else:
            self.last_reward = np.array([reward_fn(self) for reward_fn in self.reward_list], dtype=np.float32)
            if self.reward_weights is not None:
                self.last_reward = np.dot(self.last_reward, self.reward_weights)
        return self.last_reward

    # 奖励函数
    def _pressure_reward(self):
        return self.get_pressure()

    def get_pressure(self):
        return sum(
            self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes
        ) - sum(
            self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes)

    def _average_speed_reward(self):
        return self.get_average_speed()

    def get_average_speed(self) -> float:
        """车辆的平均速度，除以最大允许速度归一化"""
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += self.sumo.vehicle.getSpeed(v) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list

    def _queue_reward(self):
        return -self.get_total_queued()

    def get_total_queued(self) -> int:
        return sum(self.sumo.lane.getLastStepVehicleNumber(lane for lane in self.lanes))

    def _diff_waiting_time_reward(self):
        ts_wait = sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        reward = self.last_ts_waiting_time = ts_wait
        self.last_ts_waiting_time = ts_wait
        return reward

    def get_accumulated_waiting_time_per_lane(self):
        """每个进口车道的累计等待时间"""
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.sumo.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time_per_lane.append(wait_time)
            return wait_time_per_lane

    def _observation_fn_default(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def get_lanes_density(self) -> List[float]:
        """返回进口车道的密度（车辆数除以最大容量）。"""
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_queue(self) -> List[float]:
        """返回进口车道的排队密度（排队车辆数除以最大容量）。"""
        lanes_queue = [
            self.sumo.lane.getLastStepHaltingNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, queue) for queue in lanes_queue]

    # @classmethod
    # def register_reward_fn(cls, fn: Callable):
    #     """
    #     注册一个自定义奖励桉树到类字典中
    #     :param fn: 要注册的奖励函数
    #     :return:
    #     """
    #     if fn.__name__ in cls.reward_fns.keys():
    #         raise KeyError(f"Reward functino{fn.__name__} already exists")
    #     cls.reward_fns[fn.__name__] = fn

    # 奖励函数映射字典
    reward_fns = {
        "diff-waiting-time": _diff_waiting_time_reward,
        "average-speed": _average_speed_reward,
        "queue": _queue_reward,
        "pressure": _pressure_reward,
    }

    # if __name__ == "__main__":
    #     file_path = "../nets/osm.net.xml.gz"
    #     # tls_id = "cluster_366489708_9203769172"
    #     tls_id = "2187544212"
    #     TrafficSignal = TrafficSignal(
    #         env=,
    #         ts_id=tls_id,
    #         delta_time=1,
    #         yellow_time=5,
    #         min_green=20,
    #         max_green=120,
    #         begin_time=0,
    #         reward_fn=,
    #         reward_weights=None,
    #         sumo=,
    #     )
