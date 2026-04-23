import os
import glob
import numpy as np
import pandas as pd
import torch
from config import config
from environment.env import SumoEnvironment
from Model.DQN.DQN import Agent
from Model.vdn import VDN
from replay.multi_agent_replay_buffer import CentralizedReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(mode: str, step: int, ts_id: str) -> str:
    """根据模式、步数和路口ID返回模型文件路径"""
    model_dir = config[f"{mode}_save_path"]
    pattern = os.path.join(model_dir, ts_id, f"_{step}.pth")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No model file found: {pattern}")
    return files[0]


def demo(mode: str, step: int):
    """测试指定模式，返回包含每回合详细数据和汇总统计的字典"""
    print(f"\n========== Demo {mode.upper()} with step={step} ==========")

    env = SumoEnvironment(
        net_file=config["net_file"],
        route_file=config["route_file"],
        use_gui=True,
        begin_time=0,
        num_seconds=config["num_seconds"],
        max_depart_delay=-1,
        waiting_time_memory=1000,
        time_to_teleport=-1,
        delta_time=config["delta_time"],
        yellow_time=config["yellow_time"],
        min_green=config["min_green"],
        max_green=config["max_green"],
        single_agent=False,
        reward_fn=config["reward_fn"],
        reward_weights=config["reward_weights"],
        add_system_info=True,
        add_per_agent_info=True,
        ts_ids=config["ts_lst"],
        fixed_ts=(mode == 'fixed'),
        sumo_warnings=False,
        flag_neighbor=(mode in ["comm", "comm_ddqn"]),
        delay=50,
    )
    ts_ids = env.ts_ids
    agents = {}
    neighbors = {}
    replay_buffer = None
    vdn_trainer = None

    # 加载模型（fixed 模式不需要）
    if mode != "fixed":
        if mode in ["comm", "comm_ddqn"]:
            neighbors = {ts: env.traffic_signals[ts].neighbor for ts in ts_ids}
        for ts in ts_ids:
            state_dim = env.observation_spaces(ts).shape[0]
            action_dim = env.action_spaces(ts).n
            agents[ts] = Agent(
                ts_id=ts,
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=config["hidden_dim"],
                lr=config["lr"],
                gamma=config["gamma"],
                batch_size=config["batch_size"],
                eps_start=0.0,  # 测试时不探索
                eps_end=0.0,
                eps_decay=1,  # 无意义
                target_update=config["target_update"],
                double=(mode in ["ddqn", "vdn_ddqn", "comm_ddqn"]),
                save_path=config[f"{mode}_save_path"],
                comm_flag=(mode in ["comm", "comm_ddqn"]),
                n_agents=len(neighbors) + 1 if neighbors else 0,
                comm_embed_dim=config["comm_embed_dim"],
                neighbors=neighbors,
            )
            model_path = load_model(mode, step, ts)
            agents[ts].load_model(model_path)
            agents[ts].policy_net.eval()

        if mode in ["vdn", "vdn_ddqn", "comm", "comm_ddqn"]:
            replay_buffer = CentralizedReplayBuffer(
                capacity=1,  # 测试时不需要实际存储
                ts_ids=ts_ids,
                state_dim=env.observation_spaces(ts_ids[0]).shape[0],
                comm_dim=config["comm_embed_dim"] if mode in ["comm", "comm_ddqn"] else None,
            )
            vdn_trainer = VDN(agents, config["gamma"], config["target_update"])

    state = env.reset()
    done = {"__all__": False}

    while not done["__all__"]:
        actions = {}
        comm_vecs = {}

        # ---------- 生成动作 ----------
        if mode in ["comm", "comm_ddqn"]:
            msgs = {ts: agents[ts].encode_obs(state[ts]) for ts in ts_ids if ts in state}
            for ts in ts_ids:
                if ts not in state:
                    continue
                nbrs = neighbors.get(ts, [])
                neighbor_msgs = [msgs[nbr] for nbr in nbrs if nbr in msgs]
                comm_vecs[ts] = agents[ts].aggregate_neighbors(msgs[ts], neighbor_msgs)
            for ts in ts_ids:
                if ts not in state:
                    continue
                action = agents[ts].select_action(
                    state[ts], step=1e9, comm_vec=comm_vecs.get(ts),
                    action_mask=None, eps=0
                )
                actions[ts] = action
        elif mode in ["dqn", "ddqn", "vdn", "vdn_ddqn"]:
            for ts in ts_ids:
                if ts not in state:
                    continue
                action = agents[ts].select_action(
                    state[ts], step=1e9, comm_vec=None, action_mask=None, eps=0
                )
                actions[ts] = action
        else:  # fixed
            actions = {}

        # ---------- 环境交互 ----------
        next_state, reward, done, info = env.step(actions)
        state = next_state

    env.close()
    return


if __name__ == "__main__":
    dict = {"fixed": 0,  #
            "dqn": 143400,
            "ddqn": 143400,
            "vdn": 98400,
            "vdn_ddqn": 90600,
            "comm": 143400,
            "comm_ddqn": 143400}
    mode = "comm_ddqn"
    demo(mode, dict[mode])
