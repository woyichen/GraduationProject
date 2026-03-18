import os
import sys
import gymnasium as gym
import random
from gymnasium import spaces
import traci
import sumolib
import numpy as np
from queue import Queue


class TrafficSignal:
    def __init__(self,
                 sumo,
                 net_file: str,
                 ts_id: str,
                 yellow_time: int = 3,
                 min_green: int = 10,
                 max_green: int = 60,
                 delta_rs_update_time: int = 1,
                 use_neighbor: bool = False,
                 reward_type="delta_waiting"
                 ):
        """
        :param sumo: # 链接对象
        :param ts_id: 路口ID
        :param yellow_time: 黄灯时间
        :param min_green:最小绿灯时间
        :param max_green:最大绿灯时间
        :param delta_rs_update_time: 奖励与状态更新的时间间隔
        :param use_neighbor:是否启用邻居信息
        """
        # 仿真接口
        self.sumo = sumo
        # 路线文件
        self.net_file = net_file
        self.net = sumolib.net.readNet(self.net_file)
        # 路口ID
        self.ts_id = ts_id
        # 黄灯时间
        self.yellow_time = yellow_time
        # 绿灯时间最值及计时
        self.min_green_time = min_green
        self.max_green_time = max_green
        # 相位持续时间
        self.phase_time = 0
        # 下一次奖励/状态更新的时刻
        self.delta_rs_update_time = delta_rs_update_time
        # 上一次尝试更新动作时间
        self.last_update_time = 0

        # 当前绿灯相位对象
        self.green_phase = None
        # 当前黄灯相位对象
        self.yellow_phase = None

        # 下次可以执行动作的时间
        self.next_action_time = 0

        # 所有相位
        # self.all_phases = self.sumo.trafficlight.getAllProgramLogics(ts_id)[0].phases
        self.all_phases = None
        # 没有黄灯的相位
        # self.all_green_phases = [phase for phase in self.all_phases if 'y' not in phase.state.lower()]
        self.all_green_phases = None

        # 控制的车道
        # self.lanes = list(dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.ts_id)))
        self.lanes = None
        # 车道长度
        # self.lanes_length = {lane_id: self.sumo.lane.getLength(lane_id) for lane_id in self.lanes}
        self.lanes_length = None

        # 观测空间
        # self.observation_space = spaces.Box(
        #     low=0, high=1, shape=(len(self.lanes),), dtype=np.float32
        # )
        self.observation_space = None
        # 动作空间
        # self.action_space = spaces.Discrete(len(self.all_green_phases))
        self.action_space = None

        # 上一状态等待车辆数
        self.last_waiting = 0
        # 选择何种奖励函数
        self.reward_type = reward_type
        # 多路口通信时
        self.use_neighbor = use_neighbor
        if self.use_neighbor:
            self.neighbors = []
        if self.sumo is not None:
            self.initialize()

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
        path.add(self.ts_id)
        q = Queue()
        q.put(self.ts_id)
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

    def initialize(self):
        """
        SUMO启动后获取相位
        :return:
        """
        self.all_phases = self.sumo.trafficlight.getAllProgramLogics(self.ts_id)[0].phases
        self.all_green_phases = [p for p in self.all_phases if 'y' not in p.state.lower()]
        self.lanes = list(dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.ts_id)))
        self.lanes_length = {lane_id: self.sumo.lane.getLength(lane_id) for lane_id in self.lanes}
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.lanes),), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.all_green_phases))
        if self.use_neighbor:
            self._set_neighbors()
        # print(self.ts_id, self.all_green_phases)
        self.last_update_time = 0
        self.phase_time = 0
        self.next_action_time = 0
        self.last_waiting = 0

    def update_phase_time(self):
        """
        每个仿真步更新绿灯持续时间
        :return:
        """
        current_time = self.sumo.simulation.getTime()
        if self.last_update_time == 0:
            self.last_update_time = current_time
            return
        self.phase_time += current_time - self.last_update_time
        self.last_update_time = current_time

    def compute_state(self):
        """
        计算当前路口的状态：所有车道的密度。
        :return: 当前路口的状态向量
        """
        densities = []

        vehicle_gap = 7.5  # 5m+2.5m车辆长度+间距
        for lane in self.lanes:
            vehicle_num = self.sumo.lane.getLastStepVehicleNumber(lane)  # 当前车道车辆数
            density = min(1, vehicle_num / (self.lanes_length[lane] / vehicle_gap))  # 车道密度，最大为1
            densities.append(density)

        return np.array(densities, dtype=np.float32)

    def apply_action(self, action):
        current_time = self.sumo.simulation.getTime()
        # 未到判决时间
        if current_time < self.next_action_time:
            return

        new_phase = self.all_green_phases[action]
        # 不是黄灯时
        if self.yellow_phase is None:
            # 开始时，绿灯相位为空
            if self.green_phase is None:
                self.green_phase = new_phase
                self.phase_time = 0
                self.sumo.trafficlight.setRedYellowGreenState(self.ts_id, self.green_phase.state)
                self.next_action_time = current_time + self.delta_rs_update_time
                return
            # 未到最小绿灯时长，不转换相位
            elif self.phase_time < self.min_green_time:
                self.next_action_time = current_time + self.delta_rs_update_time
                return
            # 与当前相位相同
            elif self.green_phase.state == new_phase.state:
                # 还未达到绿灯最大时间
                if self.phase_time < self.max_green_time:
                    self.next_action_time = current_time + self.delta_rs_update_time
                    return
                # 达到绿灯时间，强制转化为下一相位
                elif self.phase_time >= self.max_green_time:
                    current_index = self.all_green_phases.index(self.green_phase)
                    next_index = (current_index + 1) % len(self.all_green_phases)
                    new_phase = self.all_green_phases[next_index]
                    yellow_state = ''
                    for i in range(len(new_phase.state)):
                        if (self.green_phase.state[i] == 'G' or self.green_phase.state[i] == 'g') and new_phase.state[
                            i] == 'r':
                            yellow_state += "y"
                        else:
                            yellow_state += self.green_phase.state[i]
                    self.yellow_phase = yellow_state
                    self.sumo.trafficlight.setRedYellowGreenState(self.ts_id, self.yellow_phase)
                    self.next_action_time = current_time + self.delta_rs_update_time
                    return
            # 下一相位与当前相位不同时
            elif self.green_phase.state != new_phase.state:
                yellow_state = ''
                for i in range(len(new_phase.state)):
                    if (self.green_phase.state[i] == 'G' or self.green_phase.state[i] == 'g') \
                            and new_phase.state[i] == 'r':
                        yellow_state += "y"
                    else:
                        yellow_state += self.green_phase.state[i]
                self.yellow_phase = yellow_state
                self.green_phase = new_phase
                self.sumo.trafficlight.setRedYellowGreenState(self.ts_id, self.yellow_phase)
                self.phase_time = 0
                self.next_action_time = current_time + self.delta_rs_update_time
                return
        # 当前为黄灯相位
        elif self.yellow_phase is not None:
            # 还未到达黄灯时长
            if self.phase_time < self.yellow_time:
                self.next_action_time = current_time + self.delta_rs_update_time
                return
            # 到达黄灯时长
            elif self.phase_time >= self.yellow_time:
                # 清除黄灯相位
                self.yellow_phase = None
                self.sumo.trafficlight.setRedYellowGreenState(self.ts_id, self.green_phase.state)
                self.phase_time = 0
                self.next_action_time = current_time + self.delta_rs_update_time
                return

    def _compute_pressure(self):
        incoming = 0
        outgoing = 0
        links = self.sumo.trafficlight.getControlledLinks(self.ts_id)
        for link_group in links:
            for link in link_group:
                in_lane = link[0]
                out_lane = link[1]

                incoming += self.sumo.lane.getLastStepVehicleNumber(in_lane)
                outgoing += self.sumo.lane.getLastStepVehicleNumber(out_lane)
        return incoming - outgoing

    def compute_reward(self):
        """
        计算路口奖励：基于车辆的等待数
        :return: 当前奖励值
        """
        # 等待车辆数
        waiting = 0
        for lane in self.lanes:
            waiting += self.sumo.lane.getLastStepHaltingNumber(lane)

        # 队列长度
        queue_length = self.get_queue_length()

        # 总等待时间
        waiting_time = self.get_total_waiting_time()

        # 平均速度
        mean_speed = self.get_mean_speed()

        # reward计算
        if self.reward_type == "queue":
            reward = -queue_length
        elif self.reward_type == "waiting":
            reward = -waiting_time
        elif self.reward_type == "delta_waiting":
            reward = self.last_waiting - waiting
            self.last_waiting = waiting
        elif self.reward_type == "mean_speed":
            reward = mean_speed
        elif self.reward_type == "pressure":
            reward = -self._compute_pressure()
        else:
            reward = self.last_waiting - waiting
            self.last_waiting = waiting
        return reward

    def get_total_waiting(self):
        """
        获取当前路口所有车道的总等待车辆数
        :return:
        """
        total = 0
        for lane in self.lanes:
            total += self.sumo.lane.getLastStepHaltingNumber(lane)
        return total

    def get_total_waiting_time(self):
        total = 0
        for lane in self.lanes:
            total += self.sumo.lane.getWaitingTime(lane)
        return total

    def get_queue_length(self):
        queue = 0
        for lane in self.lanes:
            queue += self.sumo.lane.getLastStepHaltingNumber(lane)
        return queue

    def get_mean_speed(self):
        """
        获取当前路口所有车道的速度
        :return: 平均速度
        """
        speed_num = 0
        count = 0
        for lane in self.lanes:
            v = self.sumo.lane.getLastStepMeanSpeed(lane)
            speed_num += v
            count += 1
        if count == 0:
            return 0
        return speed_num / count


if __name__ == '__main__':
    file_path = "../nets/osm.net.xml.gz"
    # tls_id = "cluster_366489708_9203769172"
    tls_id = "2187544212"
    print(tls_id)
    traci.start([sumolib.checkBinary('sumo'), '-n', file_path])
    traffic_signal = TrafficSignal(
        ts_id=tls_id,
        sumo=traci,
        net_file=file_path,
        yellow_time=2,
        min_green=5,
        max_green=60,
        delta_rs_update_time=1,
        use_neighbor=True
    )
    # for t in traffic_signal.sumo.trafficlight.getControlledLinks(traffic_signal.ts_id):
    #     print(t)
    # for lane in traffic_signal.lanes_id:
    #     print(lane)
    for t in traffic_signal.neighbors:
        print(t)
    traci.close()
    pass
