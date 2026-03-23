import os
import torch
import numpy as np
import pandas as pd
import multiprocessing as mp
# from absl import app, flags

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

    "lr": 1e-4,
    "gamma": 0.99,
    "batch_size": 32,
    "buffer_size": 50000,
    "target_update": 100,
    "seed": 42,

    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay": 100000,

    "hidden_dim": 256,
    "episodes": 800,

    "DQN_save_path": 'weights/DQN',
    "DoubleDQN_save_path": "weights/DoubleDQN",
    "MAAC_save_path": "weights/MAAC",
    "load_model": False,
    "model_path": '',

    "result_folder": "results/",
    "result_folder_name": "results",
}

# FLAGS = flags.FLAGS
# flags.DEFINE_string("net_file", 'nets/osm.net.xml.gz', "SUMO network file")
# flags.DEFINE_string("route_file", 'nets/osm.passenger.rou.xml', "SUMO route file")
# flags.DEFINE_bool("use_gui", False, "Use SUMO GUI")
#
# flags.DEFINE_integer("num_seconds", 3000, "Simulation time")
# flags.DEFINE_integer("delta_time", 5, "Action interval")
# flags.DEFINE_integer("yellow_time", 2, "Yellow phase duration")
# flags.DEFINE_integer("min_green", 10, "Minimum green time")
# flags.DEFINE_integer("max_green", 90, "Maximum green time")
#
# flags.DEFINE_float('lr', 1e-3, 'Learning rate')
# flags.DEFINE_float('gamma', 0.99, 'Discount factor')
# flags.DEFINE_integer('batch_size', 32, 'Batch size')
# flags.DEFINE_integer('buffer_size', 50000, 'Replay buffer size')
# flags.DEFINE_integer('target_update', 100, 'Target update frequency')
#
# flags.DEFINE_float('eps_start', 1.0, 'Initial epsilon')
# flags.DEFINE_float('eps_end', 0.05, 'Final epsilon')
# flags.DEFINE_float('eps_decay', 50000, 'Epsilon decay')
#
# flags.DEFINE_integer('hidden_dim', 256, 'Hidden layer size')
# # flags.DEFINE_bool('double', False, 'Use Double DQN')
#
# flags.DEFINE_integer('episodes', 200, 'Training episodes')
# flags.DEFINE_string('DQN_save_path', 'weights/DQN', 'DQN Model save path')
# flags.DEFINE_string('DoubleDQN_save_path', 'weights/DoubleDQN', 'DoubleDQN Model save path')
# flags.DEFINE_string('MAAC_save_path', 'weights/MAAC', 'MAAC Model save path')
# flags.DEFINE_bool('load_model', False, 'Load pretrained model')
# flags.DEFINE_string('model_path', '', 'Path to model')
#
# flags.DEFINE_string('result_folder', 'results/', 'Path to save results')
# flags.DEFINE_string('result_folder_name', 'results', 'Path to save results')

lst = ['2187544212', '2187544213', '2187544217', '2187544218', 'cluster_2178819374_4839352770_4839352772',
       'cluster_2178819402_2189318888', 'cluster_2187544206_4839352776', 'cluster_2187544208_4839352781',
       'cluster_366489708_9203769172']
modes = ['fixed', 'dqn', 'ddqn']


def train(mode: str, return_dict, seed: int = 42):
    """

    :param mode: ['fixed','dqn','ddqn']
    :param seed:
    :return:
    """
    # print(f"\n==Training{'DoubleDQN' if double_dqn else 'DQN'}==")
    env = SumoEnvironment(
        # net_file=FLAGS.net_file,
        # route_file=FLAGS.route_file,
        # use_gui=FLAGS.use_gui,
        net_file=config["net_file"],
        route_file=config["route_file"],
        use_gui=config["use_gui"],
        begin_time=0,
        # num_seconds=FLAGS.num_seconds,
        num_seconds=config["num_seconds"],
        max_depart_delay=-1,
        waiting_time_memory=1000,
        time_to_teleport=-1,
        # delta_time=FLAGS.delta_time,
        # yellow_time=FLAGS.yellow_time,
        # min_green=FLAGS.min_green,
        # max_green=FLAGS.max_green,
        delta_time=config["delta_time"],
        yellow_time=config["yellow_time"],
        min_green=config["min_green"],
        max_green=config["max_green"],
        single_agent=False,
        reward_fn='pressure',
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
                # hidden_dim=FLAGS.hidden_dim,
                # lr=FLAGS.lr,
                # gamma=FLAGS.gamma,
                # batch_size=FLAGS.batch_size,
                # eps_start=FLAGS.eps_start,
                # eps_end=FLAGS.eps_end,
                # eps_decay=FLAGS.eps_decay,
                # target_update=FLAGS.target_update,
                hidden_dim=config["hidden_dim"],
                lr=config["lr"],
                gamma=config["gamma"],
                batch_size=config["batch_size"],
                eps_start=config["eps_start"],
                eps_end=config["eps_end"],
                eps_decay=config["eps_decay"],
                target_update=config["target_update"],
                double=True if mode == "ddqn" else False,
                # save_path=FLAGS.DoubleDQN_save_path if mode == "ddqn" else FLAGS.DQN_save_path
                save_path=config["DoubleDQN_save_path"] if mode == "ddqn" else config["DQN_save_path"],
            )

            replay_buffers[ts] = ReplayBuffer(
                # capacity=FLAGS.buffer_size,
                capacity=config["buffer_size"],
                mode='single',
                ts_ids=ts,
                state_dim=state_dim,
                action_dim=action_dim
            )
    episode_rewards = []
    episode_speed = []
    episode_waiting = []

    global_step = 0

    # for ep in range(FLAGS.episodes):
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
        # print(f"Episode {ep} | Avg Reward: {avg_reward:.3f}")
        print(f"[{mode}] "
              f"Ep {ep} | Reward: {episode_rewards[-1]:.3f} | "
              f"Speed: {episode_speed[-1]:.3f}")
    env.close()
    result = {
        "reward": episode_rewards,
        "speed": episode_speed,
        "waiting": episode_waiting
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
    df = pd.DataFrame({
        # "episode": np.arange(FLAGS.episodes),
        "episode": np.arange(config["episodes"]),
        **{f"{k}_reward": results[k]["reward"] for k in modes},
        **{f"{k}_speed": results[k]["speed"] for k in modes},
        **{f"{k}_waiting": results[k]["waiting"] for k in modes},
    })

    # dqn_result = train(double_dqn=False, seed=0)
    # ddqn_result = train(double_dqn=True, seed=0)
    #
    # # ===== 保存结果 =====
    # df = pd.DataFrame({
    #     "episode": np.arange(FLAGS.episodes),
    #     "dqn_reward": dqn_result["reward"],
    #     "ddqn_reward": ddqn_result["reward"],
    #     "dqn_speed": dqn_result["speed"],
    #     "ddqn_speed": ddqn_result["speed"],
    #     "dqn_waiting": dqn_result["waiting"],
    #     "ddqn_waiting": ddqn_result["waiting"],
    # })

    # os.makedirs(f"{FLAGS.result_folder_name}", exist_ok=True)
    # df.to_csv(f"{FLAGS.result_folder_name}/compare.csv", index=False)
    os.makedirs(f"{config['result_folder_name']}", exist_ok=True)
    df.to_csv(f"{config['result_folder_name']}/compare.csv", index=False)

    # ===== 可视化 =====
    # draw.plot_single_metric(
    #     dqn_result["reward"],
    #     ddqn_result["reward"],
    #     title="Reward Comparison",
    #     ylabel="Average Reward",
    #     filename="reward",
    #     save_dir=FLAGS.result_folder
    # )
    #
    # draw.plot_single_metric(
    #     dqn_result["speed"],
    #     ddqn_result["speed"],
    #     title="Mean Speed Comparison",
    #     ylabel="Mean Speed",
    #     filename="speed_compare",
    #     save_dir=FLAGS.result_folder
    # )
    #
    # draw.plot_single_metric(
    #     dqn_result["waiting"],
    #     ddqn_result["waiting"],
    #     title="Total Waiting Time Comparison",
    #     ylabel="Total Waiting Time",
    #     filename="waiting_compare",
    #     save_dir=FLAGS.result_folder
    # )
    # draw.plot_compare(dqn_result, ddqn_result, save_dir=FLAGS.result_folder)

    draw.plot_multi_metric(
        {k: results[k]['reward'] for k in modes},
        title="Reward Comparison",
        ylabel="Reward",
        filename="reward",
        # save_dir=f"{FLAGS.result_folder_name}/reward",
        save_dir=f"{config['result_folder_name']}/reward",
    )
    draw.plot_multi_metric(
        {k: results[k]['speed'] for k in modes},
        title="Speed Comparison",
        ylabel="Speed",
        filename="speed",
        # save_dir=f"{FLAGS.result_folder_name}/speed",
        save_dir=f"{config['result_folder_name']}/speed",
    )
    draw.plot_multi_metric(
        {k: results[k]['waiting'] for k in modes},
        title="Waiting Time Comparison",
        ylabel="Waiting Time",
        filename="waiting_time",
        # save_dir=f"{FLAGS.result_folder_name}/waiting_time",
        save_dir=f"{config['result_folder_name']}/waiting_time",
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
