import gymnasium as gym
import os
import sys
import random

from sympy import false

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
                 net_file: str,
                 route_file: str,
                 # skip_range: int,
                 control_interval: int,
                 simulation_time: float,
                 yellow_time: int,
                 delta_rs_update_time: int,
                 use_gui: bool = false
                 ):
        self._net = net_file
        self._route = route_file
        self.control_interval = control_interval
        self.simulation_time = simulation_time
        self.yellow_time = yellow_time
        self.delta_rs_update_time = delta_rs_update_time

        self.sumoBinary = 'sumo-gui' if use_gui else 'sumo'

        traci.start([sumolib.checkBinary('sumo'), '-n', self._net])
        conn = traci
        self.ts_ids = traci.trafficlight.getIDList()
        conn.close()

