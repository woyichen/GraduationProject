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


def test(mode: str, num_episodes: int, step: int) -> dict:
    """测试指定模式，返回包含每回合详细数据和汇总统计的字典"""
    print(f"\n========== Testing {mode.upper()} with step={step} ==========")

    env = SumoEnvironment(
        net_file=config["net_file"],
        route_file=config["route_file"],
        use_gui=config["use_gui"],
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
        ts_ids=None,
        fixed_ts=(mode == 'fixed'),
        sumo_warnings=False,
        flag_neighbor=(mode in ["comm", "comm_ddqn"]),
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

    # 记录数据
    episode_rewards = []
    episode_waiting = []
    episode_speed = []
    episode_components = []  # 每回合各奖励分量（系统总和）

    for ep in range(num_episodes):
        state = env.reset()
        done = {"__all__": False}
        episode_reward = {ts: 0.0 for ts in ts_ids}
        total_reward = 0.0
        total_components = {}  # 本回合累计分量

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

            for ts in actions.keys():
                episode_reward[ts] += reward[ts]
            total_reward = sum(episode_reward.values())

            # 累加奖励分量（系统总和）
            if "reward_components" in info:
                for ts, comp_dict in info["reward_components"].items():
                    for k, v in comp_dict.items():
                        total_components[k] = total_components.get(k, 0.0) + v

            state = next_state

        # 记录本回合指标
        episode_rewards.append(total_reward)
        episode_waiting.append(info["system_total_waiting_time"])
        episode_speed.append(info["system_mean_speed"])
        episode_components.append(total_components.copy())
        print(f"  Episode {ep + 1:2d} | Reward: {total_reward:.2f} | "
              f"Waiting: {info['system_total_waiting_time']:.0f} | "
              f"Speed: {info['system_mean_speed']:.2f}")

    env.close()

    # 计算汇总统计
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_waiting = np.mean(episode_waiting)
    avg_speed = np.mean(episode_speed)

    comp_keys = config.get("reward_keys", [])
    avg_components = {}
    for k in comp_keys:
        values = [comp.get(k, 0.0) for comp in episode_components]
        avg_components[k] = np.mean(values)

    return {
        "mode": mode,
        "step": step,
        "num_episodes": num_episodes,
        "all_rewards": episode_rewards,
        "all_waiting": episode_waiting,
        "all_speed": episode_speed,
        "all_components": episode_components,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_waiting": avg_waiting,
        "avg_speed": avg_speed,
        "avg_components": avg_components,
    }


def save_test_results(results: dict, output_detail_csv: str, output_summary_csv: str):
    """使用 pandas 保存测试结果（详细数据 + 汇总数据）"""
    comp_keys = config.get("reward_keys", [])

    # ---------- 详细数据 ----------
    detail_rows = []
    for mode, res in results.items():
        for ep in range(res["num_episodes"]):
            row = {
                "mode": mode,
                "episode": ep + 1,
                "total_reward": res["all_rewards"][ep],
                "total_waiting": res["all_waiting"][ep],
                "avg_speed": res["all_speed"][ep],
            }
            comp_dict = res["all_components"][ep]
            for k in comp_keys:
                row[f"comp_{k}"] = comp_dict.get(k, 0.0)
            detail_rows.append(row)
    df_detail = pd.DataFrame(detail_rows)
    df_detail.to_csv(output_detail_csv, index=False)
    print(f"Detailed results saved to {output_detail_csv}")

    # ---------- 汇总数据 ----------
    summary_rows = []
    for mode, res in results.items():
        row = {
            "mode": mode,
            "avg_reward": res["avg_reward"],
            "std_reward": res["std_reward"],
            "avg_waiting": res["avg_waiting"],
            "avg_speed": res["avg_speed"],
        }
        for k in comp_keys:
            row[f"avg_comp_{k}"] = res["avg_components"].get(k, 0.0)
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(output_summary_csv, index=False)
    print(f"Summary results saved to {output_summary_csv}")


if __name__ == "__main__":
    test_episodes = 30
    step_interval = 100
    dict = {"fixed": 0,  #
            "dqn": 143400,
            "ddqn": 143400,
            "vdn": 98400,
            "vdn_ddqn": 90600,
            "comm": 143400,
            "comm_ddqn": 143400}
    # dict = {
    #     "fixed": 0
    # }

    for mode in dict:
        try:
            res = test(mode, num_episodes=test_episodes, step=dict[mode])
            current_result = {mode: res}
            detail_csv = f"test_results_step_{mode}_{dict[mode]}.csv"
            summary_csv = f"test_summary_step_{mode}_{dict[mode]}.csv"
            save_test_results(current_result, detail_csv, summary_csv)
        except Exception as e:
            print(f"Error testing {mode} at step {dict[mode]}: {e}")
