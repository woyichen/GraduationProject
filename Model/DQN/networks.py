import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Network(nn.Module):
    def __init__(self, inputs: int, outputs: int, hidden=512):
        super(Network, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        if isinstance(hidden, int):
            self.hidden = [hidden]
        else:
            self.hidden = hidden

        layers = []
        in_dim = self.inputs
        for h in self.hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LeakyReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, outputs))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":
    # 测试1：hidden 为整数（单隐藏层）
    print("=" * 50)
    print("测试 hidden = 64 (整数)")
    net1 = Network(inputs=10, hidden=64, outputs=4).to(device)
    print(net1)  # 打印网络结构
    x1 = torch.randn(5, 10).to(device)  # 批次大小5，输入10维
    out1 = net1(x1)
    print(f"输入形状: {x1.shape}, 输出形状: {out1.shape}")
    print()

    # 测试2：hidden 为列表（多层隐藏层）
    print("=" * 50)
    print("测试 hidden = [128, 64, 32] (多层隐藏层)")
    net2 = Network(inputs=10, hidden=[128, 64, 32], outputs=4).to(device)
    print(net2)
    x2 = torch.randn(5, 10).to(device)
    out2 = net2(x2)
    print(f"输入形状: {x2.shape}, 输出形状: {out2.shape}")
