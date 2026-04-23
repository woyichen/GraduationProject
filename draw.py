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


def draw_with_csv(csv_path, names=None):
    """
    从CSV或Excel文件读取训练数据，绘制各模型的reward曲线并保存图片。

    参数:
        csv_path (str): 输入文件路径，支持 .csv, .xlsx, .xls
        save_path (str): 输出图片保存路径（可以是文件路径或目录）
        names (list, optional): 需要绘制的模型名称列表。若为None，则自动检测所有以 '_reward' 结尾的列。
    """
    # 读取文件（自动识别扩展名）
    ext = os.path.splitext(csv_path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(csv_path)
    elif ext in ['.xlsx', '.xls']:
        try:
            df = pd.read_excel(csv_path, engine='openpyxl' if ext == '.xlsx' else 'xlrd')
        except ImportError:
            raise ImportError("请安装 openpyxl (用于 .xlsx) 或 xlrd (用于 .xls)")
    else:
        raise ValueError(f"不支持的文件类型: {ext}")

    # 确定要绘制的模型列表
    reward_cols = [col for col in df.columns if col.endswith('_reward')]
    if not reward_cols:
        raise ValueError("未找到任何以 '_reward' 结尾的列")

    if names is None:
        # 自动提取所有模型名称
        models = [col.replace('_reward', '') for col in reward_cols]
    else:
        # 只绘制 names 中指定的模型
        models = [name for name in names if f"{name}_reward" in df.columns]
        if not models:
            raise ValueError(f"指定的模型 {names} 在文件中不存在")

    # 计算各模型有效数据长度，取最小长度对齐
    lengths = []
    for model in models:
        col = f"{model}_reward"
        # 去除 NaN 后的有效行数
        valid_len = df[col].dropna().shape[0]
        lengths.append(valid_len)
    min_len = min(lengths)
    print(f"各模型数据长度: {dict(zip(models, lengths))}, 使用最小长度: {min_len}")

    # # 平滑函数（窗口5）
    # def smooth(data, w=5):
    #     if len(data) < w:
    #         return data
    #     return np.convolve(data, np.ones(w) / w, mode='valid')

    # 绘图
    plt.figure(figsize=(12, 6))
    for model in models:
        col = f"{model}_reward"
        y = df[col].dropna().values[:min_len]
        # y_smooth = smooth(y, w=5)
        x = np.arange(len(y))
        plt.plot(x, y, label=f"{model.upper()}")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Curves of Different Models")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 确定保存路径
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./results/draw", exist_ok=True)
    save_path = "./results/draw/training_curves.png"
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"曲线已保存至 {save_path}")


if __name__ == "__main__":
    csv_path = "./results/compare.csv"
    draw_with_csv(csv_path)
