import os
import torch
import numpy as np
import pandas as pd
from absl import app, flags

from environment.env import SumoEnvironment
from replay import ReplayBuffer
from Model.DQN.DQN import Agent

FLAGS = flags.FLAGS
flags.DEFINE_string("net_file", 'nets/osm.net.xml.gz', "SUMO network file")
flags.DEFINE_string("route_file", 'nets/osm.passenger.rou.xml', "SUMO route file")
flags.DEFINE_bool("use_gui", False, "Use SUMO GUI")

flags.DEFINE_integer("num_seconds", 3000, "Simulation time")
flags.DEFINE_integer("delta_time", 5, "Action interval")
flags.DEFINE_integer("yellow_time", 2, "Yellow phase duration")
flags.DEFINE_integer("min_green", 10, "Minimum green time")
flags.DEFINE_integer("max_green", 90, "Maximum green time")

flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_float('gamma', 0.99, 'Discount factor')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('buffer_size', 50000, 'Replay buffer size')
flags.DEFINE_integer('target_update', 100, 'Target update frequency')

flags.DEFINE_float('eps_start', 1.0, 'Initial epsilon')
flags.DEFINE_float('eps_end', 0.05, 'Final epsilon')
flags.DEFINE_float('eps_decay', 50000, 'Epsilon decay')

flags.DEFINE_integer('hidden_dim', 256, 'Hidden layer size')
# flags.DEFINE_bool('double', False, 'Use Double DQN')

flags.DEFINE_integer('episodes', 500, 'Training episodes')
flags.DEFINE_string('DQN_save_path', 'weights/DQN', 'DQN Model save path')
flags.DEFINE_string('DoubleDQN_save_path', 'weights/DoubleDQN', 'DoubleDQN Model save path')
flags.DEFINE_string('MAAC_save_path', 'weights/MAAC', 'MAAC Model save path')
flags.DEFINE_bool('load_model', False, 'Load pretrained model')
flags.DEFINE_string('model_path', '', 'Path to model')

flags.DEFINE_string('result_folder', 'results/', 'Path to save results')

lst = ['2187544212', '2187544213', '2187544217', '2187544218', 'cluster_2178819374_4839352770_4839352772',
       'cluster_2178819402_2189318888', 'cluster_2187544206_4839352776', 'cluster_2187544208_4839352781',
       'cluster_366489708_9203769172']


def train(double_dqn: bool, seed: int = 42):
    print(f"\n==Training{'DoubleDQN' if double_dqn else 'DQN'}==")
    env = SumoEnvironment(
        net_file=FLAGS.net_file,
        route_file=FLAGS.route_file,
        use_gui=FLAGS.use_gui,
        begin_time=0,
        num_seconds=FLAGS.num_seconds,
        max_depart_delay=-1,
        waiting_time_memory=1000,
        time_to_teleport=-1,
        delta_time=FLAGS.delta_time,
        yellow_time=FLAGS.yellow_time,
        min_green=FLAGS.min_green,
        max_green=FLAGS.max_green,
        single_agent=False,
        reward_fn='pressure',
        # reward_weights=[],
        add_system_info=True,
        add_per_agent_info=True,
        # sumo_seed=,
        ts_ids=None,
        fixed_ts=False,
        sumo_warnings=True,
        # additional_sumo_cmd=,
        # render_mode=
    )

    ts_ids = env.ts_ids

    agents = {}
    replay_buffers = {}

    for ts in ts_ids:
        state_dim = env.observation_spaces(ts).shape[0]
        action_dim = env.action_spaces(ts).n

        agents[ts] = Agent(
            ts_id=ts,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=FLAGS.hidden_dim,
            lr=FLAGS.lr,
            gamma=FLAGS.gamma,
            batch_size=FLAGS.batch_size,
            eps_start=FLAGS.eps_start,
            eps_end=FLAGS.eps_end,
            eps_decay=FLAGS.eps_decay,
            target_update=FLAGS.target_update,
            double=True if double_dqn else False,
            save_path=FLAGS.DoubleDQN_save_path if double_dqn else FLAGS.DQN_save_path
        )

        replay_buffers[ts] = ReplayBuffer(
            capacity=FLAGS.buffer_size,
            mode='single',
            ts_ids=ts,
            state_dim=state_dim,
            action_dim=action_dim
        )
    episode_rewards = []
    episode_speed = []
    episode_waiting = []

    global_step = 0

    for ep in range(FLAGS.episodes):
        state = env.reset()
        done = {"__all__": False}
        episode_reward = {ts: 0 for ts in ts_ids}

        while not done["__all__"]:

            actions = {}

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

            state = next_state

            # ===== 学习 =====
            for ts in ts_ids:
                agents[ts].learn(replay_buffers[ts])

            global_step += 1

        # ===== 输出 =====
        avg_reward = np.mean(list(episode_reward.values()))
        episode_rewards.append(avg_reward)
        episode_speed.append(info["system_mean_speed"])
        episode_waiting.append(info["system_total_waiting_time"])
        # print(f"Episode {ep} | Avg Reward: {avg_reward:.3f}")
        print(f"[{'DDQN' if double_dqn else 'DQN'}] "
              f"Ep {ep} | Reward: {avg_reward:.3f} | "
              f"Speed: {episode_speed[-1]:.3f}")
    env.close()
    return {
        "reward": episode_rewards,
        "speed": episode_speed,
        "waiting": episode_waiting
    }


from matplotlib import pyplot as plt


def plot_single_metric(dqn, ddqn, title, ylabel, filename, save_dir="results", smooth_w=5):
    os.makedirs(save_dir, exist_ok=True)

    def smooth(x, w):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode='valid')

    x_dqn = smooth(dqn, smooth_w)
    x_ddqn = smooth(ddqn, smooth_w)

    plt.figure()
    plt.plot(x_dqn, label="DQN")
    plt.plot(x_ddqn, label="Double DQN")

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()

    # ===== 保存 =====
    png_path = os.path.join(save_dir, f"{FLAGS.result_folder}{filename}.png")
    # pdf_path = os.path.join(save_dir, f"{FLAGS.result_folder}{filename}.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    # plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()  # 防止内存堆积


def plot_compare(dqn_res, ddqn_res):
    def smooth(x, w=5):
        return np.convolve(x, np.ones(w) / w, mode='valid')

    plt.figure(figsize=(12, 8))

    # ===== Reward =====
    plt.subplot(3, 1, 1)
    plt.plot(smooth(dqn_res["reward"]), label="DQN")
    plt.plot(smooth(ddqn_res["reward"]), label="Double DQN")
    plt.title("Reward")
    plt.legend()
    plt.grid()

    # ===== Speed =====
    plt.subplot(3, 1, 2)
    plt.plot(smooth(dqn_res["speed"]), label="DQN")
    plt.plot(smooth(ddqn_res["speed"]), label="Double DQN")
    plt.title("Mean Speed")
    plt.legend()
    plt.grid()

    # ===== Waiting =====
    plt.subplot(3, 1, 3)
    plt.plot(smooth(dqn_res["waiting"]), label="DQN")
    plt.plot(smooth(ddqn_res["waiting"]), label="Double DQN")
    plt.title("Total Waiting Time")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f'{FLAGS.result_folder}_.png')
    plt.show()


def main(argv):
    del argv

    dqn_result = train(double_dqn=False, seed=0)
    ddqn_result = train(double_dqn=True, seed=0)

    # ===== 保存结果 =====
    df = pd.DataFrame({
        "episode": np.arange(FLAGS.episodes),
        "dqn_reward": dqn_result["reward"],
        "ddqn_reward": ddqn_result["reward"],
        "dqn_speed": dqn_result["speed"],
        "ddqn_speed": ddqn_result["speed"],
        "dqn_waiting": dqn_result["waiting"],
        "ddqn_waiting": ddqn_result["waiting"],
    })

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/dqn_vs_ddqn.csv", index=False)

    # ===== 可视化 =====
    plot_single_metric(
        dqn_result["reward"],
        ddqn_result["reward"],
        title="Reward Comparison",
        ylabel="Average Reward",
        filename="reward_compare"
    )

    plot_single_metric(
        dqn_result["speed"],
        ddqn_result["speed"],
        title="Mean Speed Comparison",
        ylabel="Mean Speed",
        filename="speed_compare"
    )

    plot_single_metric(
        dqn_result["waiting"],
        ddqn_result["waiting"],
        title="Total Waiting Time Comparison",
        ylabel="Total Waiting Time",
        filename="waiting_compare"
    )
    plot_compare(dqn_result, ddqn_result)


if __name__ == "__main__":
    app.run(main)
