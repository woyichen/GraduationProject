import torch
import torch.nn as nn


def check_cuda_info():
    print("=== CUDA信息检查 ===")

    # 基本可用性
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA版本: {torch.version.cuda}")

    if torch.cuda.is_available():
        # 设备信息
        device_count = torch.cuda.device_count()
        print(f"CUDA设备数量: {device_count}")

        for i in range(device_count):
            print(f"\n--- 设备 {i} ---")
            print(f"名称: {torch.cuda.get_device_name(i)}")

            props = torch.cuda.get_device_properties(i)
            print(f"计算能力: {props.major}.{props.minor}")
            print(f"显存: {props.total_memory / 1024 ** 3:.2f} GB")

        # 当前设备
        print(f"\n当前设备索引: {torch.cuda.current_device()}")

        # 创建测试张量
        tensor = torch.randn(2, 3).cuda()
        print(f"\n测试张量设备: {tensor.device}")
        print(f"是否在CUDA上: {tensor.is_cuda}")


# check_cuda_info()
