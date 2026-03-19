"""
delta_time动作间隔>yellow_time黄灯时间
"""

import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the old_environment variable 'SUMO_HOME'")
import gymnasium as gym
import numpy as np
import pandas as pd
import sumolib
import traci
from gymnasium.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

try:
    # pettingzoo 1.25+
    from pettingzoo.utils import AgentSelector
except ImportError:
    # pettingzoo 1.24 or earlier
    from pettingzoo.utils import agent_selector as AgentSelector

from pettingzoo.utils.conversions import parallel_wrapper_fn

from environment.observations import DefaultObservationFunction, ObservationFunction
from environment.traffic_signal import TrafficSignal

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


def env(**kwargs):
    """PettingZoo环境"""
    env = SumoEnvironmentPZ(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class SumoEnvironment(gym.Env):
    """SUMO交通信号控制环境

    Args:
        net_file (str): SUMO路网文件
        route_file (str): SUMO路线文件
        out_csv_name (Optional[str]): 输出.csv文件名，None时不输出
        use_gui (bool): 是否使用GUI运行仿真
        virtual_display (Optional[Tuple[int,int]]): 虚拟显示的分辨率
        begin_time (int): 仿真开始时间 Default:20000
        num_seconds (int): 仿真总时长 Default: 20000
        max_depart_delay (int): 车辆插入的最大延迟时间，超时丢弃 Default: -1 (no delay)
        waiting_time_memory (int): 车辆等待时间的记忆时长 Default: 1000
        time_to_teleport (int): 车辆卡住多少秒后将其传送到路段末端 Default: -1 (no teleport)
        delta_time (int): 动作时间间隔 Default: 5 seconds
        yellow_time (int): 黄灯相位时长 Default: 2 seconds
        min_green (int): 绿灯相位最小时长 Default: 5 seconds
        max_green (int): 绿灯相位最大时长 Default: 60 seconds. Warning: This parameter is currently ignored!
        single_agent (bool): True为单个智能体
        reward_fn (str/function/dict/List): 奖励函数。字符串、字典{信号灯ID：奖励函数}、列表[]
        reward_weights (List[float]/np.ndarray): reward_fn为列表时，线性组合奖励的权重，None返回多维奖励数组 Default: None
        observation_class (ObservationFunction): 观测函数类
        add_system_info (bool): 为True，在info字典中添加系统级指标（总排队、总等待时间、平均速度等）
        add_per_agent_info (bool): 为True，在info字典中提娜佳每个智能体的指标
        sumo_seed (int/string): SUMO随机种子，为"random"时使用随机种子
        ts_ids (Optional[List[str]]): 需要控制的交通信号灯ID列表。为None控制仿真所有交通信号灯
        fixed_ts (bool): 为True，则按照route_file中的相位配置运行，忽略step方法中传入的动作
        sumo_warnings (bool): 为True，打印SUMO警告信息
        additional_sumo_cmd (str): 额外的SUMO命令行参数
        render_mode (str): 渲染模式，'human'或'rgb_array' Default: None
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    CONNECTION_LABEL = 0  # traci的多客户端连接，每创建一个环境实例自增

    def __init__(
            self,
            net_file: str,
            route_file: str,
            out_csv_name: Optional[str] = None,
            use_gui: bool = False,
            virtual_display: Tuple[int, int] = (3200, 1800),
            begin_time: int = 0,
            num_seconds: int = 20000,
            max_depart_delay: int = -1,
            waiting_time_memory: int = 1000,
            time_to_teleport: int = -1,
            delta_time: int = 5,
            yellow_time: int = 2,
            min_green: int = 5,
            max_green: int = 50,
            single_agent: bool = False,
            reward_fn: Union[str, Callable, dict, List] = "diff-waiting-time",
            reward_weights: Optional[List[float]] = None,
            observation_class: type[ObservationFunction] = DefaultObservationFunction,
            add_system_info: bool = True,
            add_per_agent_info: bool = True,
            sumo_seed: Union[str, int] = "random",
            ts_ids: Optional[List[str]] = None,
            fixed_ts: bool = False,
            sumo_warnings: bool = True,
            additional_sumo_cmd: Optional[str] = None,
            render_mode: Optional[str] = None,
    ) -> None:
        assert render_mode is None or render_mode in self.metadata["render_modes"], "Invalid render mode."
        self.render_mode = render_mode
        self.virtual_display = virtual_display
        self.disp = None

        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        if self.use_gui or self.render_mode is not None:
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")

        assert delta_time > yellow_time, "Time between actions must be at least greater than yellow time."
        assert max_green > min_green, "Max green time must be greater than min green time."

        self.begin_time = begin_time
        self.sim_max_time = begin_time + num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.waiting_time_memory = waiting_time_memory  # Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime)
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.single_agent = single_agent
        self.reward_fn = reward_fn
        self.reward_weights = reward_weights
        self.sumo_seed = sumo_seed
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.additional_sumo_cmd = additional_sumo_cmd
        self.add_system_info = add_system_info
        self.add_per_agent_info = add_per_agent_info
        self.label = str(SumoEnvironment.CONNECTION_LABEL)
        SumoEnvironment.CONNECTION_LABEL += 1
        self.sumo = None

        # 临时连接以获取交通信号灯信息
        if LIBSUMO:
            print('LIBSUMO is enabled.')
            traci.start(
                [sumolib.checkBinary("sumo"), "-n", self._net])  # Start only to retrieve traffic light information
            conn = traci
        else:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net], label="init_connection" + self.label)
            conn = traci.getConnection("init_connection" + self.label)

        if ts_ids is None:
            self.ts_ids = list(conn.trafficlight.getIDList())
        else:
            self.ts_ids = ts_ids
        self.observation_class = observation_class

        self._build_traffic_signals(conn)

        conn.close()

        # 存储车辆等待时间历史
        self.vehicles = dict()
        # 奖励范围无限
        self.reward_range = (-float("inf"), float("inf"))
        self.episode = 0
        # 存储每步信号
        self.metrics = []
        self.out_csv_name = out_csv_name
        # 每个智能体的观测
        self.observations = {ts: None for ts in self.ts_ids}
        # 每个智能体的奖励
        self.rewards = {ts: None for ts in self.ts_ids}

    def _build_traffic_signals(self, conn):
        """
        每个信号灯对应的TrafficSignal
        :param conn:
        :return:
        """
        if not isinstance(self.reward_fn, dict):
            self.reward_fn = {ts: self.reward_fn for ts in self.ts_ids}

        self.traffic_signals = {
            ts: TrafficSignal(
                self,
                ts,
                self.delta_time,
                self.yellow_time,
                self.min_green,
                self.max_green,
                self.begin_time,
                self.reward_fn[ts],
                self.reward_weights,
                conn,
            )
            for ts in self.ts_ids
        }

    def _start_simulation(self):
        """
        启动SUMO仿真
        :return:
        """
        sumo_cmd = [
            self._sumo_binary,
            "-n",
            self._net,
            "-r",
            self._route,
            "--max-depart-delay",
            str(self.max_depart_delay),
            "--waiting-time-memory",
            str(self.waiting_time_memory),
            "--time-to-teleport",
            str(self.time_to_teleport),
        ]
        if self.begin_time > 0:
            sumo_cmd.append(f"-b {self.begin_time}")
        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")
        if self.additional_sumo_cmd is not None:
            sumo_cmd.extend(self.additional_sumo_cmd.split())
        if self.use_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])
            if self.render_mode == "rgb_array":
                sumo_cmd.extend(["--window-size", f"{self.virtual_display[0]},{self.virtual_display[1]}"])
                from pyvirtualdisplay.smartdisplay import SmartDisplay

                print("Creating a virtual display.")
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()
                print("Virtual display started.")

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

        if self.use_gui or self.render_mode is not None:
            if "DEFAULT_VIEW" not in dir(traci.gui):  # traci.gui.DEFAULT_VIEW is not defined in libsumo
                traci.gui.DEFAULT_VIEW = "View #0"
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    def reset(self, seed: Optional[int] = None, **kwargs):
        """重置环境"""
        super().reset(seed=seed, **kwargs)

        # 不是第一轮，关闭旧仿真并保存CSV
        if self.episode != 0:
            self.close()
            self.save_csv(self.out_csv_name, self.episode)
        self.episode += 1
        self.metrics = []

        if seed is not None:
            self.sumo_seed = seed
        self._start_simulation()  # 启动仿真

        # 重新构建交通信号对象
        self._build_traffic_signals(self.sumo)

        # 重置
        self.vehicles = dict()
        self.num_arrived_vehicles = 0
        self.num_departed_vehicles = 0
        self.num_teleported_vehicles = 0

        # 计算初始观测
        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]], self._compute_info()
        else:
            return self._compute_observations()

    @property
    def sim_step(self) -> float:
        """返回当前仿真时间"""
        return self.sumo.simulation.getTime()

    def step(self, action: Union[dict, int]):
        """
        执行动作，将仿真向前推进 delta_time 秒。
        :param action: 单智能体：整数；多智能体：字典
        :return:
        """
        # 固定配时/无动作，不执行动作
        if self.fixed_ts or action is None or action == {}:
            for _ in range(self.delta_time):
                self._sumo_step()
        else:
            self._apply_actions(action)
            self._run_steps()

        # 计算新的观测、奖励、终止标志和信息
        observations = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        terminated = False  # 没有终止状态，只有截断
        truncated = dones["__all__"]  # 仿真时间达到最大值时截断
        info = self._compute_info()

        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], terminated, truncated, info
        else:
            return observations, rewards, dones, info

    def _run_steps(self):
        """
        仿真直到下一个决策时刻
        :return:
        """
        time_to_act = False
        while not time_to_act:
            self._sumo_step()
            for ts in self.ts_ids:
                self.traffic_signals[ts].update()
                # 某个智能体需要决策
                if self.traffic_signals[ts].time_to_act:
                    time_to_act = True

    def _apply_actions(self, actions):
        """
        应用动作
        :param actions: 单智能体为：整数；多智能体为：{ts_id:相位索引}
        :return:
        """
        if self.single_agent:
            if self.traffic_signals[self.ts_ids[0]].time_to_act:
                self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
        else:
            for ts, action in actions.items():
                if self.traffic_signals[ts].time_to_act:
                    self.traffic_signals[ts].set_next_phase(action)

    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.ts_ids}
        dones["__all__"] = self.sim_step >= self.sim_max_time
        return dones

    def _compute_info(self):
        info = {"step": self.sim_step}
        if self.add_system_info:
            # 添加信息
            info.update(self._get_system_info())
        if self.add_per_agent_info:
            # 添加每个智能体的信息
            info.update(self._get_per_agent_info())
        self.metrics.append(info.copy())
        return info

    def _compute_observations(self):
        """
        计算观测值
        :return:
        """
        # 计算观测并更新
        self.observations.update(
            {
                ts: self.traffic_signals[ts].compute_observation()
                for ts in self.ts_ids
                if self.traffic_signals[ts].time_to_act or self.fixed_ts
            }
        )
        return {
            ts: self.observations[ts].copy()
            for ts in self.observations.keys()
            if self.traffic_signals[ts].time_to_act or self.fixed_ts
        }

    def _compute_rewards(self):
        self.rewards.update(
            {
                ts: self.traffic_signals[ts].compute_reward()
                for ts in self.ts_ids
                if self.traffic_signals[ts].time_to_act or self.fixed_ts
            }
        )
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if
                self.traffic_signals[ts].time_to_act or self.fixed_ts}

    @property
    def observation_space(self):
        """
        返回单智能体的观测空间
        """
        return self.traffic_signals[self.ts_ids[0]].observation_space

    @property
    def action_space(self):
        """
        返回单智能体的动作空间
        """
        return self.traffic_signals[self.ts_ids[0]].action_space

    @property
    def reward_space(self):
        """
        返回单智能体的奖励空间
        """
        return self.traffic_signals[self.ts_ids[0]].reward_space

    @property
    def reward_dim(self):
        """
        返回单智能体的奖励维度
        """
        return self.traffic_signals[self.ts_ids[0]].reward_dim

    def observation_spaces(self, ts_id: str):
        """返回指定智能体的观测空间"""
        return self.traffic_signals[ts_id].observation_space

    def action_spaces(self, ts_id: str) -> gym.spaces.Discrete:
        """返回指定智能体的动作空间"""
        return self.traffic_signals[ts_id].action_space

    def _sumo_step(self):
        """
        执行一步SUMO仿真
        """
        self.sumo.simulationStep()
        self.num_arrived_vehicles += self.sumo.simulation.getArrivedNumber()
        self.num_departed_vehicles += self.sumo.simulation.getDepartedNumber()
        self.num_teleported_vehicles += self.sumo.simulation.getEndingTeleportNumber()

    def _get_system_info(self):
        """
        获取系统信息，包括当前运行车辆数、排队车辆数、等待时间灯
        """
        # 所有车辆ID
        vehicles = self.sumo.vehicle.getIDList()
        # 各车速度
        speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        # 各车等待时间
        waiting_times = [self.sumo.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
        # 待插入车辆数
        num_backlogged_vehicles = len(self.sumo.simulation.getPendingVehicles())
        return {
            # 正在运行的车辆数
            "system_total_running": len(vehicles),
            # 待插入车辆数
            "system_total_backlogged": num_backlogged_vehicles,
            # 停止的车辆数
            "system_total_stopped": sum(
                int(speed < 0.1) for speed in speeds
            ),  # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            # 累计到达车辆数
            "system_total_arrived": self.num_arrived_vehicles,
            # 累计出发车辆数
            "system_total_departed": self.num_departed_vehicles,
            # 累计传送车辆数
            "system_total_teleported": self.num_teleported_vehicles,
            # 总等待时间
            "system_total_waiting_time": sum(waiting_times),
            # 平均等待时间
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            # 平均速度
            "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
        }

    def _get_per_agent_info(self):
        """
        获取每个智能体的指标，计算所有智能体的总和
        :return: 排队数、累积等待时间、平均速度
        """
        # 每个信号灯控制区域内的排队车辆
        stopped = [self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids]
        # 每个信号灯控制区域的累计等待时间
        accumulated_waiting_time = [
            sum(self.traffic_signals[ts].get_accumulated_waiting_time_per_lane()) for ts in self.ts_ids
        ]
        # 每个区域内的累计等待时间
        average_speed = [self.traffic_signals[ts].get_average_speed() for ts in self.ts_ids]
        info = {}
        for i, ts in enumerate(self.ts_ids):
            # 智能体的排队数
            info[f"{ts}_stopped"] = stopped[i]
            # 智能体的累计等待时间
            info[f"{ts}_accumulated_waiting_time"] = accumulated_waiting_time[i]
            # 智能体的平均速度
            info[f"{ts}_average_speed"] = average_speed[i]
        # 所有智能体的排队总数
        info["agents_total_stopped"] = sum(stopped)
        # 所有智能体的累计等待时间总和
        info["agents_total_accumulated_waiting_time"] = sum(accumulated_waiting_time)
        return info

    def close(self):
        """关闭环境"""
        if self.sumo is None:
            return

        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()

        if self.disp is not None:
            self.disp.stop()
            self.disp = None

        self.sumo = None

    def __del__(self):
        self.close()

    def render(self):
        """ 渲染当前仿真画面。

        若 render_mode 为 "human"，则 SUMO GUI 已显示窗口，无需额外操作。
        若 render_mode 为 "rgb_array"，则从虚拟显示器截取当前画面并返回 numpy 数组。
        """
        if self.render_mode == "human":
            return  # 已经渲染
        elif self.render_mode == "rgb_array":
            # img = self.sumo.gui.screenshot(traci.gui.DEFAULT_VIEW,
            #                          f"temp/img{self.sim_step}.jpg",
            #                          width=self.virtual_display[0],
            #                          height=self.virtual_display[1])
            img = self.disp.grab()
            return np.array(img)

    def save_csv(self, out_csv_name, episode):
        """
        保存记录的指标
        :param out_csv_name: 输出文件名
        :param episode: episode编号
        :return:
        """
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + f"_conn{self.label}_ep{episode}" + ".csv", index=False)

    # 用于表格型方法

    def encode(self, state, ts_id):
        """将连续观测状态编码为可哈希的元组，用于离散状态空间。"""
        phase = int(np.where(state[: self.traffic_signals[ts_id].num_green_phases] == 1)[0])
        min_green = state[self.traffic_signals[ts_id].num_green_phases]
        density_queue = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases + 1:]]
        # tuples are hashable and can be used as key in python dictionary
        return tuple([phase, min_green] + density_queue)

    def _discretize_density(self, density):
        """
        将密度值[0,1]离散为0-9的整数
        :param density:
        :return:
        """
        return min(int(density * 10), 9)


class SumoEnvironmentPZ(AECEnv, EzPickle):
    """
    SUMO 环境的 PettingZoo AEC 接口包装器。

    该类实现了 PettingZoo 的 AECEnv 接口，使环境可以用于多智能体强化学习库（如 PettingZoo）。
    参数与 SumoEnvironment 相同。
    """

    metadata = {"render.modes": ["human", "rgb_array"], "name": "sumo_rl_v0", "is_parallelizable": True}

    def __init__(self, **kwargs):
        # 用于序列化
        EzPickle.__init__(self, **kwargs)
        # 保存参数字典
        self._kwargs = kwargs

        # 初始化随机数生成器
        self.seed()
        # 创建内部SUMO环境
        self.env = SumoEnvironment(**self._kwargs)
        # 渲染模式
        self.render_mode = self.env.render_mode

        # 当前智能体列表
        self.agents = self.env.ts_ids
        # 所有可能的智能体
        self.possible_agents = self.env.ts_ids
        # 智能体选择器
        self._agent_selector = AgentSelector(self.agents)
        # 重置选择器，获得第一个智能体
        self.agent_selection = self._agent_selector.reset()
        # 为每个智能体存储其动作空间和观测空间
        self.action_spaces = {a: self.env.action_spaces(a) for a in self.agents}
        self.observation_spaces = {a: self.env.observation_spaces(a) for a in self.agents}

        # 初始化字典
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def seed(self, seed=None):
        """设置环境的随机种子"""
        self.randomizer, seed = seeding.np_random(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境，开始新的一轮"""
        # 重置内部环境
        self.env.reset(seed=seed, options=options)
        # 恢复所有智能体
        self.agents = self.possible_agents[:]
        # 重置选择器
        self.agent_selection = self._agent_selector.reset()
        # 清空奖励
        self.rewards = {agent: 0 for agent in self.agents}
        # 累计奖励清零
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        # 终止标志复位
        self.terminations = {a: False for a in self.agents}
        # 阶段标志复位
        self.truncations = {a: False for a in self.agents}
        # 计算初试信息
        self.compute_info()

    def compute_info(self):
        """
        从内部环境获取信息，并分配到每个智能体的info字典中
        :return:
        """
        self.infos = {a: {} for a in self.agents}
        infos = self.env._compute_info()
        for a in self.agents:
            for k, v in infos.items():
                if k.startswith(a) or k.startswith("system"):
                    self.infos[a][k] = v

    def observation_space(self, agent):
        """返回指定智能体的观测空间"""
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """返回指定智能体的观动作空间"""
        return self.action_spaces[agent]

    def observe(self, agent):
        """返回指定智能体的当前观测值"""
        obs = self.env.observations[agent].copy()
        return obs

    def close(self):
        """关闭环境"""
        self.env.close()

    def render(self):
        """渲染当前画面"""
        return self.env.render()

    def save_csv(self, out_csv_name, episode):
        """保存仿真指标到CSV文件"""
        self.env.save_csv(out_csv_name, episode)

    def step(self, action):
        """执行当前选定的智能体的动作，并推进环境"""
        if self.truncations[self.agent_selection] or self.terminations[self.agent_selection]:
            return self._was_dead_step(action)
        agent = self.agent_selection
        if not self.action_spaces[agent].contains(action):
            raise Exception(
                f"Action for agent {agent} must be in Discrete({self.action_spaces[agent].n}).\n",
                f"It is currently {action}"
            )

        if not self.env.fixed_ts:
            self.env._apply_actions({agent: action})

        if self._agent_selector.is_last():
            if not self.env.fixed_ts:
                self.env._run_steps()
            else:
                for _ in range(self.env.delta_time):
                    self.env._sumo_step()

            self.env._compute_observations()
            self.rewards = self.env._compute_rewards()
            self.compute_info()
        else:
            self._clear_rewards()

        done = self.env._compute_dones()["__all__"]
        self.truncations = {a: done for a in self.agents}

        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        # 累积奖励
        self._accumulate_rewards()
