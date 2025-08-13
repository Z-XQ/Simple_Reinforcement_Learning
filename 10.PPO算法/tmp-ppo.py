import gymnasium as gym
import torch
import torch.nn as nn


"""
Actor: p = actor_net(s)
Critic: v = critic_net(s) 状态价值
policy_loss: loss = -min(rA, clip(r)A), A = A*0.98*0.95 + delta, delta = r+0.98*critic_net(s') - critic_net(s)
critic_loss: loss = mse(r+critic_net(s') - critic_net(s))
"""

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
            state: tensor. (1, 4)

        Returns: prob. tensor. (1, 2)

        """
        return self.layer(state)


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
            state: tensor. (1, 4)

        Returns: tensor. state value. (1, 1)

        """
        return self.layer(state)

