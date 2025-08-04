import math
import os
import random
from collections import namedtuple, deque

import gymnasium as gym
import torch
import torch.nn as nn


Transition = namedtuple(typename="Transition", field_names="state, action, reward, next_state")


class ReplayMemory(object):

    def __init__(self, n=1500):
        self.memory = deque([], maxlen=n)

    def push(self, *args):
        """

        Args:
            *args: state, action, reward, next_state

        Returns:

        """
        transition = Transition(*args)
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """近似Q函数，输入s，输出q值=net(s)"""
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_states, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.layer(x)


class CartPoleTrainer(object):
    def __init__(self, model_path="models/123.pth"):
        # 1. 检查是否使用 GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)} with {torch.cuda.device_count()} GPUs")
        else:
            print("Using CPU")

        # 2. 设置环境
        self.env = gym.make('CartPole-v1')

        # 3. 双网络初始化（策略网络+目标网络）
        n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.policy_net = DQN(n_states, self.n_actions)
        self.target_net = DQN(n_states, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 4. 设置优化器(amsgrad稳定学习率，避免过早衰减)与经验回放缓冲区
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=1e-4, amsgrad=True)
        self.memory = ReplayMemory(10000)

        # 5. 超参数定义
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99  # 折扣因子（未来奖励的衰减系数）
        self.TARGET_UPDATE_RATE = 10  # 目标网络更新频率（每10个episode更新一次）

        # eps-greedy
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.step_done = 0

        self.model_path = model_path
        save_dir = os.path.dirname(model_path)
        os.makedirs(save_dir, exist_ok=True)

    def select_action_by_eps_greedy(self, state):
        """eps-greedy: 大概率采用q值最大的action，小概率采取其他动作"""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        # eps
        eps = self.EPS_END + (self.EPS_START-self.EPS_END) * math.exp(-1.0 * self.step_done / self.EPS_DECAY)
        #
        self.step_done += 1
        p = random.random()

        if p > eps:
            with torch.no_grad():
                q_val = self.policy_net(state_tensor)  # (b,2)
                action = torch.argmax(q_val, dim=1).item()
        else:
            action = random.choices(range(self.n_actions), k=1)[0]
        return action



