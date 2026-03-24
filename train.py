import numpy as np
from config import config
from environment.env import SumoEnvironment
from replay.replay_buffer import ReplayBuffer
from replay.multi_agent_replay_buffer import CentralizedReplayBuffer
from Model.DQN.DQN import Agent
from Model.vdn import VDN


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

        reward_fn=config["reward_fn"],
        reward_weights=config["reward_weights"],

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


def train_vdn(mode: str, return_dict, seed: int = 42):
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
        fixed_ts=False,
        sumo_warnings=False,
        # additional_sumo_cmd=,
        # render_mode=
    )
    ts_ids = env.ts_ids

    agents = {}
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
            double=False,
            save_path=config["VDN_save_path"],
        )

    replay_buffer = CentralizedReplayBuffer(
        capacity=config["buffer_size"],
        ts_ids=ts_ids,
    )

    vdn_trainer = VDN(agents, config["gamma"], config["target_update"])

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
            for ts in ts_ids:
                if ts not in state:
                    continue
                action = agents[ts].select_action(state[ts], global_step)
                actions[ts] = action

            next_state, reward, done, info = env.step(actions)

            if not env.fixed_ts:
                replay_buffer.add(state, actions, next_state, reward, done)
                for ts in actions.keys():
                    episode_reward[ts] += reward[ts]

            vdn_trainer.update(replay_buffer, config["batch_size"])
            global_step += 1
            state = next_state

        episode_rewards.append(np.mean(list(episode_reward.values())))
        episode_waiting.append(info["system_total_waiting_time"])
        episode_speed.append(info["system_mean_speed"])
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

if __name__ == "__main__":
    train_vdn()
