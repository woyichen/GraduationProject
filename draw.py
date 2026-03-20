import os
import numpy as np
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
    folder = filename + '_' + 'results'
    png_path = os.path.join(folder, f"{filename}.png")
    # pdf_path = os.path.join(save_dir, f"{FLAGS.result_folder}{filename}.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    # plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()  # 防止内存堆积


def plot_compare(dqn_res, ddqn_res, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

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
    plt.savefig(os.path.join(save_dir, '_.png'))
    plt.show()
