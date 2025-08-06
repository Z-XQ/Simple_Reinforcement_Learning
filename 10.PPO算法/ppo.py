import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# 设置随机种子，保证结果可复现
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


# 定义Actor-Critic网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()

        # 共享的特征提取层
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # 策略网络 (Actor) - 输出动作概率分布
        self.actor = nn.Linear(hidden_dim, action_dim)

        # 价值网络 (Critic) - 估计状态价值
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = self.shared_layer(state)  # (4, ) -> (64, )
        logits = self.actor(x)  # (64, ) -> (2, )
        value = self.critic(x)  # (64, ) -> (1, )
        return logits, value

    def get_action(self, state):
        # 获得动作和对应的对数概率、状态价值
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)  # 对于离散动作空间使用分类分布
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.item()


# 定义PPO算法
class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=3e-4,
                 gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
                 K_epochs=10, entropy_coef=0.01, value_loss_coef=0.5):
        self.gamma = gamma  # 折扣因子
        self.gae_lambda = gae_lambda  # GAE系数
        self.clip_epsilon = clip_epsilon  # PPO截断系数
        self.K_epochs = K_epochs  # 每批数据更新次数
        self.entropy_coef = entropy_coef  # 熵正则化系数
        self.value_loss_coef = value_loss_coef  # 价值损失系数

        # 初始化Actor-Critic网络
        self.model = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def compute_gae(self, rewards, values, dones, next_value):
        """计算广义优势估计(GAE)"""
        advantages = []
        returns = []
        advantage = 0

        # 从最后一步向前计算
        for t in reversed(range(len(rewards))):
            # 如果是终止状态，下一个状态的价值为0
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.gae_lambda * advantage * (1 - dones[t])
            next_value = values[t]

            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[t])  # 回报 = 优势 + 价值

        # 标准化优势，提高训练稳定性
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self, states, actions, old_log_probs, advantages, returns):
        """使用PPO-Clip更新策略"""
        # 将数据转换为PyTorch张量
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        # 多次更新同一批数据
        for _ in range(self.K_epochs):
            # 计算当前策略的logits和价值
            logits, values = self.model(states)
            dist = Categorical(logits=logits)

            # 计算当前策略的对数概率和熵
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # 计算新旧策略的概率比值
            ratio = torch.exp(log_probs - old_log_probs)

            # 计算PPO-Clip目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 计算价值损失
            critic_loss = F.mse_loss(values.squeeze(), returns)

            # 总损失 = 策略损失 + 价值损失 - 熵正则化(鼓励探索)
            total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

            # 反向传播和优化
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()


def train_ppo(env_name="CartPole-v1", max_episodes=300, max_timesteps=1000,
              batch_size=512, print_interval=10):
    # 创建环境
    env = gym.make(env_name)
    # env.seed(seed)

    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 初始化PPO代理
    ppo = PPO(state_dim, action_dim)

    # 记录训练过程中的奖励
    scores = []
    # 用于计算最近100个episode的平均奖励
    scores_window = deque(maxlen=100)

    # 主训练循环
    for episode in range(1, max_episodes + 1):
        state = env.reset()
        if isinstance(state, tuple):  # 处理gym 0.26+的返回格式变化
            state = state[0]

        # 存储一个batch的数据
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        score = 0

        for t in range(max_timesteps):
            # 选择动作
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action, log_prob, value = ppo.model.get_action(state_tensor)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储数据
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(1 if done else 0)

            # 更新状态和分数
            state = next_state
            score += reward

            # 如果达到batch_size或者 episode结束，更新策略
            if len(states) >= batch_size or done:
                # 获取最后一个状态的价值
                with torch.no_grad():
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
                    _, next_value = ppo.model(next_state_tensor)
                    next_value = next_value.item()

                # 计算GAE和回报
                advantages, returns = ppo.compute_gae(rewards, values, dones, next_value)

                # 更新PPO
                ppo.update(states, actions, log_probs, advantages, returns)

                # 重置存储的batch数据
                states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

            if done:
                break

        # 记录奖励
        scores.append(score)
        scores_window.append(score)

        # 打印训练进度
        if episode % print_interval == 0:
            print(f"Episode {episode}\tAverage Score: {np.mean(scores_window):.2f}")

        # 如果连续100个episode的平均奖励达到475，认为解决了问题
        if np.mean(scores_window) >= 475:
            print(f"\nEnvironment solved in {episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}")
            torch.save(ppo.model.state_dict(), "ppo_cartpole.pth")
            break

    env.close()
    return scores


# 训练并绘制结果
scores = train_ppo(max_episodes=3000)

# 绘制奖励曲线
plt.figure(figsize=(10, 6))
plt.plot(scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.title('PPO-Clip on CartPole-v1')

# 绘制滑动平均曲线
window_size = 10
smoothed_scores = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
plt.plot(range(window_size - 1, len(scores)), smoothed_scores, 'r-', linewidth=2)
plt.legend(['Episode Score', f'{window_size}-Episode Moving Average'])
plt.show()
