import random
import gymnasium as gym
import torch
import torch.nn as nn
from tqdm import tqdm


#定义模型
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, state):
        """
        Args:
            state: tensor. (4, )
        Returns:
            actor: p = actor_net(s)
        """
        p = self.layer(state)  # (4, ) -> (2, )
        return p


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        """
        Args:
            state: tensor. (4, )
        Returns:
            actor: val = net(s)
        """
        p = self.layer(state)  # (4, ) -> (1, )
        return p


class CartPoleTrainer(object):
    def __init__(self):
        self.env = gym.make("CartPole-v1")

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

    # 优势函数
    @staticmethod
    def get_advantages(deltas):
        advantages = []

        # 反向遍历deltas
        s = 0.0
        for delta in deltas[::-1]:
            s = 0.98 * 0.95 * s + delta
            advantages.append(s)

        # 逆序
        advantages.reverse()
        return advantages