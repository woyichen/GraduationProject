import os
import pandas as pd

import numpy as np
from matplotlib import pyplot as plt


def plot_multi_metric(df, modes, keys, title, save_dir, smooth_w=5):
    os.makedirs(save_dir, exist_ok=True)

    def smooth(x, w):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode='valid')

    for key in keys:
        plt.figure()
        for mode in modes:
            smoothed = smooth(df[f'{mode}_{key}'], smooth_w)
            plt.plot(smoothed, label=mode.upper())
        plt.title(f'{title}_{key}')
        plt.xlabel("Episode")
        plt.ylabel(f'{key}')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, f'{title}_{key}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def draw_with_csv(csv_path, save_path, names):
    pass
