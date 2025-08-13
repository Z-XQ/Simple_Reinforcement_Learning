import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

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


class CartPoleTrainer(object):
    def __init__(self):
        # env
        self.env = gym.make("CartPole-v1")

        # model
        state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.actor_net = Actor(state_dim, self.action_dim)
        self.critic_net = Critic(state_dim)

        self.optimizer_actor = torch.optim.Adam(self.actor_net.parameters(), lr=1e-3)
        self.optimizer_critic = torch.optim.Adam(self.critic_net.parameters(), lr=1e-2)
        self.loss_fn = torch.nn.MSELoss()

        # 超参数
        self.BATCH_SIZE = 128

    def get_action(self, state):
        """

        Args:
            state: array. (4, )

        Returns:

        """
        state_t = torch.tensor(state, dtype=torch.float32).reshape(1, 4)
        prob = self.actor_net(state_t)
        action = random.choices(range(self.action_dim), weights=prob[0].tolist(), k=1)[0]
        return action

    def get_data(self):
        """
        Returns:
            states: [b, 4]
            rewards: [b, 1]
            actions: [b, 1]
            next_states: [b, 4]
            overs: [b, 1]
        """
        states, actions, rewards, overs, next_states = [], [], [], [], []
        over = False
        state, _ = self.env.reset()
        while not over:
            action = self.get_action(state)
            next_state, reward, terminate, truncated, info = self.env.step(action)
            over = terminate or truncated
            # 记录
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            overs.append(over)
            next_states.append(next_state)
            # next
            state = next_state
        # tensor
        states = torch.tensor(np.array(states), dtype=torch.float32).reshape(-1, 4)  # [b, 4]
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).reshape(-1, 1)  # [b, 1]
        actions = torch.tensor(np.array(actions), dtype=torch.long).reshape(-1, 1)  # [b, 1]
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).reshape(-1, 4)  # [b, 4]
        overs = torch.tensor(np.array(overs), dtype=torch.long).reshape(-1, 1)  # [b, 1]
        return states, actions, rewards, overs, next_states


    def train(self):

        for epoch in tqdm(range(1500)):
            # 1. 获取batch data
            states, actions, rewards, overs, next_states = self.get_data()

            # 2. target error
            # values
            values = self.critic_net(states)  # (128,4) -> (128,1)

            # TD target
            target_values = self.critic_net(next_states)  # (128,4) -> (128,1)
            target_values *= (1 - overs)
            target_values = rewards + 0.98*target_values
            target_values = target_values.detach()

            # target error: delta
            deltas = target_values - values  # (128, 1) error = r + net(s') - net(s)

            # 3. advantage
            advantages = self.get_advantage(deltas)  # (128, 1) -> list of float
            advantages = torch.tensor(advantages, dtype=torch.float32).reshape(-1, 1)  # (128,1)

            # 4. 取出每一步动作的概率
            old_probs = self.actor_net(states)  # [b, 2]
            old_probs = old_probs.gather(dim=1, index=actions)  # actions.shape=[b, 1]
            old_probs = old_probs.detach()

            # 多次迭代，逼近目标
            for i in range(10):
                # 4. policy loss
                # 求出概率的变化ratio
                new_probs = self.actor_net(states)  # (128,4) -> (128,2)
                new_probs = new_probs.gather(dim=1, index=actions)  # (128,1)
                ratios = new_probs / old_probs  # (128,1)

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
                loss_actor = -torch.min(surr1, surr2)  # (128,1)
                loss_actor = loss_actor.mean()  #

                # 5. value loss
                # values
                vals = self.critic_net(states)  # (128,4) -> (128,1)
                # target values
                vals_loss = self.loss_fn(vals, target_values)

                self.optimizer_actor.zero_grad()
                loss_actor.backward()
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                vals_loss.backward()
                self.optimizer_critic.step()

            if epoch % 50 == 0:
                test_result = sum([self.test() for _ in range(10)]) / 10
                print(epoch, test_result)

    def test(self):
        # 初始化游戏
        state, _ = self.env.reset()

        # 记录反馈值的和,这个值越大越好
        reward_sum = 0

        # 玩到游戏结束为止
        over = False
        while not over:
            # 根据当前状态得到一个动作
            # action = self.get_action_by_max_p(state)
            action = self.get_action(state)
            # 执行动作,得到反馈
            state, reward, terminate, truncated, _ = self.env.step(action)
            over = terminate or truncated
            reward_sum += reward

        return reward_sum

    def get_advantage(self, deltas):
        """

        Args:
            deltas: tensor. (128,1)

        Returns: list of float.

        """
        advantages = []

        s = 0
        for i in reversed(range(deltas.shape[0])):
            s = 0.98*0.95*s + deltas[i]
            advantages.append(s)

        advantages.reverse()
        return advantages


if __name__ == '__main__':
    trainer = CartPoleTrainer()
    trainer.train()



