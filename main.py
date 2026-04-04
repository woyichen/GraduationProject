import os

# os.environ["LIBSUMO_AS_TRACI"] = "1"
# sumo_bin = r"D:\School\Eclipse\Sumo\bin"  # 替换为你的实际路径
# if os.path.exists(sumo_bin):
#     os.add_dll_directory(sumo_bin)
import numpy as np
import pandas as pd
import multiprocessing as mp

import draw
from train import train
from config import config


def main():
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    for mode in config["modes"]:
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
        for mode in config["modes"]:
            comps = results[mode]['reward_components'][ep]
            for key in config["reward_keys"]:
                row.append(comps[key])
        rows.append(row)
    columns = ['episode']
    for mode in config["modes"]:
        for key in config["reward_keys"]:
            columns.append(f"{mode}_{key}")
    rewards_df = pd.DataFrame(rows, columns=columns)
    rewards_df.to_csv(f"{config['result_folder_name']}/rewards.csv", index=False)

    df = pd.DataFrame({
        "episode": np.arange(config["episodes"]),
        **{f"{k}_reward": results[k]["reward"] for k in config["modes"]},
        **{f"{k}_speed": results[k]["speed"] for k in config["modes"]},
        **{f"{k}_waiting": results[k]["waiting"] for k in config["modes"]},
    })

    df.to_csv(f"{config['result_folder_name']}/compare.csv", index=False)

    draw.plot_multi_metric(
        df=df,
        modes=config["modes"],
        keys=['reward', 'speed', 'waiting'],
        title="sys",
        save_dir=f"{config['result_folder_name']}/sys",
    )
    draw.plot_multi_metric(
        df=rewards_df,
        modes=config["modes"],
        keys=config["reward_keys"],
        title="reward",
        save_dir=f"{config['result_folder_name']}/reward",
    )


if __name__ == "__main__":
    main()
