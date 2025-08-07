from collections import deque

import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Categorical

import numpy as np
import torch.nn.functional as F


# ac网络模型
class ActorCritic(nn.Module):
    """
    Actor: p=net(s) 输出动作概率分布
    Critic: v=net(s) 输出状态价值
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        # 共享的特征提取层
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        # actor
        self.actor = nn.Linear(hidden_dim, action_dim)  # (64,) -> (2,)
        # critic
        self.critic = nn.Linear(hidden_dim, 1)  # (64,) -> (1,)

    def forward(self, state):
        """
        Actor: p=net(s) 输出动作概率分布
        Critic: v=net(s) 输出状态价值
        Args:
            state: tensor. (4,)
        Returns:
            logits: tensor. (2,). raw prob. 所有动作的概率
            value: tensor. (1,). 状态价值v(s)
        """
        x = self.shared_layer(state)  # (4,) -> (64,)
        logits_p = self.actor(x)  # (64,) -> (2,)
        s_value = self.critic(x)    # (64,) -> (1,)
        return logits_p, s_value

    def get_action(self, state):
        """
        概率高的动作被选中的可能性更大，同时保留一定的探索性。
        eps-greedy: 若ε=0.1，则 90% 概率选最优动作，10% 概率随机选任意动作（假设有 2 个动作，则每个随机动作概率 5%）。
        Categorical 分布采样: 若动作 A 的概率是 0.7，动作 B 是 0.3，则采样时 A 有 70% 概率被选中，B 有 30% 概率被选中，不存在 “完全随机” 的独立分支。
        Args:
            state: tensor. (4,)
        Returns:
            action: int. 0/1
            log_prob(a): float. log_prob. 选取动作对应的log概率
            value: float. v(s) 状态价值
        """
        # 1. value
        logits, s_value = self.forward(state)

        # 2. action, log_prob
        distribution = Categorical(logits=logits)  # 内部会用"softmax"将logits转为概率. 对象。
        action = distribution.sample()  # 从这个分布中采样一个动作（按照概率分布，不是直接选最大概率）
        log_prob = distribution.log_prob(action)  # 计算这个采样动作在当前分布下的对数概率
        return action.item(), log_prob, s_value.item()


class CartPoleTrainer(object):
    def __init__(self):
        # 1. env
        self.env = gym.make("CartPole-v1")

        # 2. model
        n_states = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.ac_model = ActorCritic(n_states, n_actions, hidden_dim=64)
        self.optimizer = torch.optim.Adam(self.ac_model.parameters(), lr=3e-4)

        # 3. 超参数
        # 计算gae
        self.gamma = 0.99        # 0.99 折扣因子
        self.gae_lambda = 0.95   # 0.95 GAE系数，平衡偏差和方差

        # loss: policy_loss + 0.5*value_loss + 0.01H
        self.clip_epsilon = 0.2      # 0.2 PPO截断系数，(1-eps, 1+eps). 用于策略损失函数
        self.entropy_coef = 0.01     # 熵正则化系数 用于随机损失
        self.value_loss_coef = 0.5   # 价值损失系数

        self.k_epochs = 10  # 同属于一个episode数据迭代多次。
        self.batch_size = 512

    def compute_gae(self, rewards, values, dones, next_value):
        """
        计算广义优势估计(GAE)、或者折扣回报也行。
        GAE：TD误差的加权和, 平衡偏差与方差。
        输入: 同一个episode数据
        Args:
            rewards: list of float. 时间步t到T-1的即时奖励列表（[r_t, r_{t+1}, ..., r_{T-1}]）
            values:  list of float. 时间步t到T-1的状态价值（[V(s_t), V(s_{t+1}), ..., V(s_{T-1})]）
            dones:   list of int. 标记每个状态是否为终止状态（[done_t, done_{t+1}, ..., done_{T-1}]，list. 终止时为 1，否则为 0）
            next_value: float. 最后一个状态s_T的价值估计（V(s_T)， float32. 若s_T是终止状态则为 0）
        Returns:
            advantages：array. (T,). 每个时间步的优势估计（[A_t, A_{t+1}, ..., A_{T-1}]）
            returns：list. (T,). 每个时间步的回报估计（[R_t, R_{t+1}, ..., R_{T-1}]，即Q(s_t,a_t)的估计）
        """
        advantages, returns = [], []

        advantage = 0  # 不同时刻的优势价值，最后一个状态的优势价值为0
        for t in reversed(range(len(rewards))):
            # 1. 计算TD误差（Temporal Difference Error）: delta_t = r_t + γ * V(s_{t+1}) * (1-done_t) - V(s_t)
            delta_t = rewards[t] + self.gamma*next_value*(1-dones[t]) - values[t]
            next_value = values[t]

            # 2. 累积优势（GAE的核心：TD误差的加权和）: A_t = delta_t + γ*λ * A_{t+1} * (1-done_t)
            advantage = delta_t + self.gamma*self.gae_lambda*advantage*(1-dones[t])

            # 3. R_t = A_t + V(s_t)（因为优势的定义A_t = Q_t - V(s_t)，而Q_t是动作价值理论值，R_t可近似动作价值）
            # 因为是基于当前单条轨迹数据计算的，所以计算出的是此条轨迹的return值，而不是return值的期望值即动作价值吗。
            ret_val = advantage + values[t]

            # 4. 将当前优势和回报插入列表开头（保持时间顺序）
            advantages.insert(0, advantage)
            returns.insert(0, ret_val)
            # 5. 标准化优势，提高训练稳定性（可选但重要，避免不同量级的优势干扰训练）

        # 标准化优势，提高训练稳定性（可选但重要，避免不同量级的优势干扰训练）
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def train(self):
        # 记录训练过程中的奖励
        scores = []  # 打印
        scores_window = deque(maxlen=100)  # 打印 用于计算最近100个episode的平均奖励

        # 主训练循环
        for episode in range(3000):
            state, _ = self.env.reset()

            # 存储同属于一个episode的一个batch的数据
            states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
            score_episode = 0  # 打印
            done = False
            while not done:
                # 选择动作，此动作的log_prob，此状态价值
                state_tensor = torch.tensor(state, dtype=torch.float32)  # tensor(4,)
                action, log_prob, value = self.ac_model.get_action(state_tensor)  # actor网络给出动作、及其对应的概率、此状态价值

                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # 存储数据
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)
                dones.append(1 if done else 0)

                # 如果达到batch_size或者episode结束，更新策略
                if len(states) >= self.batch_size or done:  # episode很短也不能浪费，也要训练。一开始都很短。
                    self.train_episode(states, actions, log_probs, rewards, values, dones, next_state)
                    states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []  # 重置存储的batch数据
                else:
                    pass  # 不清空，即当前episode没有结束或者没有达到batch大小，可以继续添加

                # next 更新状态和分数
                state = next_state
                score_episode += reward
                if done:
                    break

            # 记录奖励
            scores.append(score_episode)
            scores_window.append(score_episode)

            # 打印训练进度
            if episode % 10 == 0:
                print(f"Episode {episode}\tAverage Score: {np.mean(scores_window):.2f}")

            # 如果连续100个episode的平均奖励达到475，认为解决了问题
            if np.mean(scores_window) >= 475:
                print(
                    f"\nEnvironment solved in {episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}")
                torch.save(self.ac_model.state_dict(), "ppo_cartpole.pth")
                break

    def train_episode(self, states, actions, log_probs, rewards, values, dones, next_state):
        """
        训练同属于一个episode的一个batch的数据
        Args:
            states: list of array.
            actions: list of int.
            log_probs: list of float.
            rewards:  list of float.
            values: list of float. 状态价值
            dones: list of int. 可能是刚好够batch数据，最后一个状态是未结束状态，此时依然可以用于计算
            next_state: float. 最后一个状态
        Returns:

        """
        # 获取最后一个状态的价值
        with torch.no_grad():
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            _, next_value = self.ac_model(next_state_tensor)  # 评价网络，给出评价值-估计此状态价值
            next_value = next_value.item()

        # 计算GAE和回报：计算同一个episode中不同时刻的gae和回报值
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        # 更新PPO：一个episode中不同时刻的状态、动作、此动作概率、gae和回报值
        self.update(states, actions, log_probs, advantages, returns)

    def update(self, states, actions, old_log_probs, advantages, returns):
        """使用PPO-Clip更新策略
        一个episode中不同时刻的状态、动作、此动作概率、gae和回报值
        """
        # 将数据转换为PyTorch张量
        states = torch.tensor(states, dtype=torch.float32)  # list of array(4,) -> (18,4)
        actions = torch.tensor(actions, dtype=torch.long)  # list of int. -> (18,)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)  # list of tensor(0) -> (18,)
        advantages = torch.tensor(advantages, dtype=torch.float32)  # array(18,) -> (18, )
        returns = torch.tensor(returns, dtype=torch.float32)  # list of float32. -> (18, )

        # 多次更新同一批数据（同一个episode）
        for _ in range(self.k_epochs):
            # 计算当前策略的logits和价值
            logits, values = self.ac_model(states)  # states(18,4) logits(18,2) values(18,1)
            dist = Categorical(logits=logits)  # 内部会用"softmax"将logits转为概率.

            # 计算当前策略的对数概率和熵
            log_probs = dist.log_prob(actions)  # actions(18,) log_probs(18,)
            entropy = dist.entropy().mean()  # tensor(0.6)

            # 计算新旧策略的概率比值e^p_new / e^p_old
            ratio = torch.exp(log_probs - old_log_probs)  # (18,)

            # 计算PPO-Clip目标
            surr1 = ratio * advantages  # surr1(18,)
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages  # surr2(18,)
            actor_loss = -torch.min(surr1, surr2).mean()  # tensor(-0.0000004)

            # 计算价值损失: critic评估的状态价值values(18,1), 实际的returns值(18,)，最小化这两者的差值
            critic_loss = F.mse_loss(values.squeeze(), returns)  # tensor(58.7536)

            # 总损失 = 策略损失 + 0.5*价值损失 - 0.01*熵正则化(鼓励探索)
            total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

            # 反向传播和优化
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()


if __name__ == '__main__':
    trainer = CartPoleTrainer()
    trainer.train()