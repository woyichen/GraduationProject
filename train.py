import time
import numpy as np
from config import config
from environment.env import SumoEnvironment
from replay.replay_buffer import ReplayBuffer
from replay.multi_agent_replay_buffer import CentralizedReplayBuffer
from Model.DQN.DQN import Agent
from Model.vdn import VDN


def train(mode: str, return_dict, seed: int = 42):
    """

    :param mode: ['fixed','dqn','ddqn','vdn','comm']
    :param seed:
    :return:
    """
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
        flag_neighbor=True if mode == "comm" or mode == "comm_ddqn" else False,
    )
    ts_ids = env.ts_ids
    neighbors = {}
    agents = {}
    replay_buffers = {}
    if mode != 'fixed':
        for ts in ts_ids:
            if mode in ["comm", "comm_ddqn"]:
                neighbors[ts] = env.traffic_signals[ts].neighbor
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
                double=True if mode in ["ddqn", "vdn_ddqn", "comm_ddqn"] else False,
                save_path=config[f"{mode}_save_path"],
                comm_flag=True if mode in ["comm", "comm_ddqn"] else False,
                n_agents=len(neighbors) + 1 if neighbors is not None else 0,
                comm_embed_dim=config["comm_embed_dim"],
                neighbors=neighbors,
            )
            if mode in ["dqn", "ddqn"]:
                replay_buffers[ts] = ReplayBuffer(
                    capacity=config["buffer_size"],
                    mode='single',
                    ts_ids=ts,
                    state_dim=state_dim,
                    action_dim=action_dim
                )
    if mode in ["comm", "comm_ddqn"]:
        replay_buffer = CentralizedReplayBuffer(
            capacity=config["buffer_size"],
            ts_ids=ts_ids,
            state_dim=env.observation_spaces(ts_ids[0]).shape[0],
            comm_dim=config["comm_embed_dim"],
        )
        vdn_trainer = VDN(agents, config["gamma"], config["target_update"], neighbors=neighbors)
    elif mode in ["vdn", "vdn_ddqn"]:
        replay_buffer = CentralizedReplayBuffer(
            capacity=config["buffer_size"],
            ts_ids=ts_ids,
            state_dim=env.observation_spaces(ts_ids[0]).shape[0],
            # comm_dim=config["comm_embed_dim"],
        )
        vdn_trainer = VDN(agents, config["gamma"], config["target_update"])
    episode_rewards = []
    episode_waiting = []
    episode_speed = []
    episode_all_rewards = []

    global_step = 0

    for ep in range(config["episodes"]):
        state = env.reset()
        sim_start_time = env.sim_step
        real_start_time = time.time()
        done = {"__all__": False}
        episode_reward = {ts: 0 for ts in ts_ids}

        while not done["__all__"]:
            actions = {}
            comm_vecs = {}
            next_comm_vecs = {}
            if mode in ["comm", "comm_ddqn"]:
                msgs = {ts: agents[ts].encode_obs(state[ts]) for ts in ts_ids}
                # comm_vecs = {}
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
                        state[ts], global_step, comm_vec=comm_vecs.get(ts),
                        action_mask=None
                    )
                    actions[ts] = action

            if mode in ["dqn", "ddqn", "vdn", "vdn_ddqn", "comm", "comm_ddqn"]:
                # ===== 选择动作 =====
                for ts in ts_ids:
                    # 注意：SUMO-RL只在time_to_act时提供state
                    if ts not in state:
                        continue

                    action = agents[ts].select_action(
                        state[ts],
                        global_step,
                        comm_vec=comm_vecs.get(ts),
                        action_mask=None
                    )
                    actions[ts] = action

            # ===== 执行动作 =====
            next_state, reward, done, info = env.step(actions)

            if mode in ["comm", "comm_ddqn"] and not done["__all__"]:
                next_msgs = {ts: agents[ts].encode_obs(next_state[ts]) for ts in ts_ids if ts in next_state}
                for ts in ts_ids:
                    if ts not in next_state:
                        continue
                    nbrs = neighbors.get(ts, [])
                    neighbor_msgs = [next_msgs[nbr] for nbr in nbrs if nbr in next_msgs]
                    next_comm_vecs[ts] = agents[ts].aggregate_neighbors(next_msgs[ts], neighbor_msgs)

            if mode != "fixed":
                # ===== 存储经验 =====
                if mode in ["dqn", "ddqn"]:
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
                elif mode in ["vdn", "vdn_ddqn", "comm", "comm_ddqn"]:
                    replay_buffer.add(state, actions, next_state, reward, done,
                                      comm=comm_vecs, next_comm=next_comm_vecs)
                    for ts in actions.keys():
                        episode_reward[ts] += reward[ts]

                # ===== 学习 =====
                if mode in ["dqn", "ddqn"]:
                    for ts in ts_ids:
                        agents[ts].learn(replay_buffers[ts])
                    global_step += 1
                elif mode in ["vdn", "vdn_ddqn", "comm", "comm_ddqn"]:
                    vdn_trainer.update(replay_buffer, config["batch_size"])
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
        sim_end_time = env.sim_step
        real_end_time = time.time()
        print(f"[{mode}] "
              f"Ep {ep} | Reward: {episode_rewards[-1]:.3f} | "
              f"SimTime: {sim_end_time - sim_start_time:.0f}s | RealTime: {real_end_time - real_start_time:.0f}s")
    env.close()
    result = {
        "reward": episode_rewards,
        "speed": episode_speed,
        "waiting": episode_waiting,
        "reward_components": episode_all_rewards,
    }
    return_dict[mode] = result
    return result
