import random
import torch
import torch.nn as nn
import torch.optim as optim

from Model.DQN.networks import Network


def DQN(state_size, action_size, seed):
    torch.manual_seed(seed)
