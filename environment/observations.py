"""交通信号观测函数"""

from abc import abstractmethod
import numpy as np
from gymnasium import spaces
from environment.traffic_signal import TrafficSignal


class ObservationFunction:
    """抽象基类"""

    def __init__(self, ts: TrafficSignal):
        """初始化观测函数"""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """计算当前观测值"""
        pass

    @abstractmethod
    def observation_space(self):
        """返回观测空间的规格"""
        pass


class DefaultObservationFunction(ObservationFunction):
    """交通信号的默认观测函数"""

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """
        1：当前绿灯独热编码
        2：是否已过最小绿灯时间
        3：进口车道的密度列表
        4：获取所有进口车道的排队长度
        :return:np.array(1+2+3+4)
        """
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """绿灯相位数量+最小绿灯标志+2*进口车道数量（密度+排队）"""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )
