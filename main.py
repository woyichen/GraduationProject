import os
import numpy as np
import pandas as pd
import multiprocessing as mp

from environment.env import SumoEnvironment
from replay.replay_buffer import ReplayBuffer
from Model.DQN.DQN import Agent
import draw

config = {
    "net_file": "nets/osm.net.xml.gz",
    "route_file": 'nets/osm.passenger.rou.xml',
    "use_gui": False,
    "num_seconds": 3000,
    "delta_time": 5,
    "yellow_time": 2,
    "min_green": 10,
    "max_green": 90,

    "lr": 5e-3,
    "gamma": 0.99,
    "batch_size": 32,
    "buffer_size": 50000,
    "target_update": 100,
    "seed": 42,

    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay": 100000,

    "hidden_dim": 256,
    "episodes": 2,

    "DQN_save_path": 'weights/DQN',
    "DoubleDQN_save_path": "weights/DoubleDQN",
    "MAAC_save_path": "weights/MAAC",
    "load_model": False,
    "model_path": '',

    "result_folder": "results/",
    "result_folder_name": "results",
}

lst = ['2187544212', '2187544213', '2187544217', '2187544218', 'cluster_2178819374_4839352770_4839352772',
       'cluster_2178819402_2189318888', 'cluster_2187544206_4839352776', 'cluster_2187544208_4839352781',
       'cluster_366489708_9203769172']
modes = ['fixed', 'dqn', 'ddqn']
reward_keys = ['diff-waiting-time', 'average-speed', 'queue', 'pressure']


def train(mode: str, return_dict, seed: int = 42):
    """

    :param mode: ['fixed','dqn','ddqn']
    :param seed:
    :return:
    """
    # print(f"\n==Training{'DoubleDQN' if double_dqn else 'DQN'}==")
    env = SumoEnvironment(
        net_file=config["net_file"],
        route_file=config["route_file"],
        use_gui=config["use_gui"],
        begin_time=0,
        # num_seconds=FLAGS.num_seconds,
        num_seconds=config["num_seconds"],
        max_depart_delay=-1,
        waiting_time_memory=1000,
        time_to_teleport=-1,
        delta_time=config["delta_time"],
        yellow_time=config["yellow_time"],
        min_green=config["min_green"],
        max_green=config["max_green"],
        single_agent=False,

        reward_fn='queue',
        # reward_weights=[],

        add_system_info=True,
        add_per_agent_info=True,
        # sumo_seed=,
        ts_ids=None,
        fixed_ts=(mode == 'fixed'),
        sumo_warnings=False,
        # additional_sumo_cmd=,
        # render_mode=
    )
    ts_ids = env.ts_ids

    agents = {}
    replay_buffers = {}
    if mode != 'fixed':
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
                eps_start=config["eps_start"],
                eps_end=config["eps_end"],
                eps_decay=config["eps_decay"],
                target_update=config["target_update"],
                double=True if mode == "ddqn" else False,
                save_path=config["DoubleDQN_save_path"] if mode == "ddqn" else config["DQN_save_path"],
            )

            replay_buffers[ts] = ReplayBuffer(
                capacity=config["buffer_size"],
                mode='single',
                ts_ids=ts,
                state_dim=state_dim,
                action_dim=action_dim
            )
    episode_rewards = []
    episode_waiting = []
    episode_speed = []
    episode_all_rewards = []

    global_step = 0

    for ep in range(config["episodes"]):
        state = env.reset()
        done = {"__all__": False}
        episode_reward = {ts: 0 for ts in ts_ids}

        while not done["__all__"]:

            actions = {}
            if mode != "fixed":
                # ===== 选择动作 =====
                for ts in ts_ids:
                    # 注意：SUMO-RL只在time_to_act时提供state
                    if ts not in state:
                        continue

                    action = agents[ts].select_action(
                        state[ts],
                        global_step
                    )

                    actions[ts] = action

            # ===== 执行动作 =====
            next_state, reward, done, info = env.step(actions)

            if mode != "fixed":
                # ===== 存储经验 =====
                for ts in actions.keys():
                    if ts not in next_state:
                        continue

                    replay_buffers[ts].add(
                        state[ts],
                        actions[ts],
                        next_state[ts],
                        reward[ts],
                        done["__all__"],
                        None  # 如果未来支持mask，在此传入
                    )

                    episode_reward[ts] += reward[ts]

                # ===== 学习 =====
                for ts in ts_ids:
                    agents[ts].learn(replay_buffers[ts])

                global_step += 1
            else:
                for ts in ts_ids:
                    if ts in reward:
                        episode_reward[ts] += reward[ts]

            state = next_state

        # ===== 输出 =====
        episode_rewards.append(np.mean(list(episode_reward.values())))
        episode_speed.append(info["system_mean_speed"])
        episode_waiting.append(info["system_total_waiting_time"])
        components = info["reward_components"]
        agg = {}
        for ts in components:
            for k, v in components[ts].items():
                agg.setdefault(k, []).append(v)
        agg = {k: np.mean(v) for k, v in agg.items()}
        episode_all_rewards.append(agg)
        # print(f"Episode {ep} | Avg Reward: {avg_reward:.3f}")
        print(f"[{mode}] "
              f"Ep {ep} | Reward: {episode_rewards[-1]:.3f} | "
              f"Speed: {episode_speed[-1]:.3f}")
    env.close()
    result = {
        "reward": episode_rewards,
        "speed": episode_speed,
        "waiting": episode_waiting,
        "reward_components": episode_all_rewards,
    }
    return_dict[mode] = result
    return result


# def main(argv):
#     del argv
def main():
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    for mode in modes:
        p = mp.Process(target=train, args=(mode, return_dict))
        p.start()
        print(f"Spawned process PID:{p.pid}")
        processes.append(p)
    for p in processes:
        p.join()

    results = dict(return_dict)
    # print(results)
    os.makedirs(f"{config['result_folder_name']}", exist_ok=True)
    rows = []
    for ep in range(config["episodes"]):
        row = [ep]
        for mode in modes:
            comps = results[mode]['reward_components'][ep]
            for key in reward_keys:
                row.append(comps[key])
        rows.append(row)
    columns = ['episode']
    for mode in modes:
        for key in reward_keys:
            columns.append(f"{mode}_{key}")
    rewards_df = pd.DataFrame(rows, columns=columns)
    rewards_df.to_csv(f"{config['result_folder_name']}/rewards.csv", index=False)

    df = pd.DataFrame({
        # "episode": np.arange(FLAGS.episodes),
        "episode": np.arange(config["episodes"]),
        **{f"{k}_reward": results[k]["reward"] for k in modes},
        **{f"{k}_speed": results[k]["speed"] for k in modes},
        **{f"{k}_waiting": results[k]["waiting"] for k in modes},
    })

    df.to_csv(f"{config['result_folder_name']}/compare.csv", index=False)

    draw.plot_multi_metric(
        df=df,
        modes=modes,
        keys=['reward', 'speed', 'waiting'],
        title="sys",
        save_dir=f"{config['result_folder_name']}/sys",
    )
    draw.plot_multi_metric(
        df=rewards_df,
        modes=modes,
        keys=reward_keys,
        title="reward",
        save_dir=f"{config['result_folder_name']}/reward",
    )


if __name__ == "__main__":
    # app.run(main)
    main()

    # # 读取 CSV 文件
    # df = pd.read_csv(f"{config['result_folder_name']}/compare.csv")
    #
    # # 获取所有列名
    # columns = df.columns.tolist()
    #
    # # 提取所有模式名（通过 _reward 后缀）
    # modes = set()
    # for col in columns:
    #     if col.endswith('_reward'):
    #         mode = col[:-7]  # 去掉 "_reward"
    #         modes.add(mode)
    #
    # # 重构 results
    # results = {}
    # for mode in modes:
    #     results[mode] = {
    #         'reward': df[f"{mode}_reward"].tolist(),
    #         'speed': df[f"{mode}_speed"].tolist(),
    #         'waiting': df[f"{mode}_waiting"].tolist()
    #     }
    #
    # draw.plot_multi_metric(
    #     {k: results[k]['reward'] for k in modes},
    #     title="Reward Comparison",
    #     ylabel="Reward",
    #     filename="reward",
    #     # save_dir=f"{FLAGS.result_folder_name}/reward",
    #     save_dir=f"{config['result_folder_name']}/reward",
    # )
    # draw.plot_multi_metric(
    #     {k: results[k]['speed'] for k in modes},
    #     title="Speed Comparison",
    #     ylabel="Speed",
    #     filename="speed",
    #     # save_dir=f"{FLAGS.result_folder_name}/speed",
    #     save_dir=f"{config['result_folder_name']}/speed",
    # )
    # draw.plot_multi_metric(
    #     {k: results[k]['waiting'] for k in modes},
    #     title="Waiting Time Comparison",
    #     ylabel="Waiting Time",
    #     filename="waiting_time",
    #     # save_dir=f"{FLAGS.result_folder_name}/waiting_time",
    #     save_dir=f"{config['result_folder_name']}/waiting_time",
    # )
