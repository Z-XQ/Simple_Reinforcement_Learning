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

        # 策略网络 (Actor) - 输出动作概率分布
        self.actor = nn.Linear(hidden_dim, action_dim)

        # 价值网络 (Critic) - 估计状态价值
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        Actor: p=net(s) 输出动作概率分布
        Critic: v=net(s) 输出状态价值
        Args:
            state: tensor. (4,).代表某个状态s
        Returns:
            logits: tensor. (2,). raw prob. 所有动作的概率
            value: tensor. (1,). 状态价值v(s)
        """
        x = self.shared_layer(state)  # (4, ) -> (64, )
        logits = self.actor(x)  # (64, ) -> (2, ) 未归一化的动作概率分数（也叫 “对数几率”）
        value = self.critic(x)  # (64, ) -> (1, ) 状态价值估计，v(s)
        return logits, value

    def get_action(self, state):
        """
        概率高的动作被选中的可能性更大，同时保留一定的探索性。
        Args:
            state: tensor. (4, ). 输入状态s
        Returns:
            action: int. 0/1
            action: float. log_prob.
            value: float. v(s) 状态价值
        """
        # 获得动作和对应的对数概率、状态价值
        logits, value = self.forward(state)

        # 2. 基于logits创建一个分类概率分布（离散动作空间专用）
        dist = Categorical(logits=logits)  # 内部会用"softmax"将logits转为概率. 对象。
        # 3. 从这个分布中采样一个动作（不是直接选最大概率）
        action = dist.sample()  # tensor(0)
        # 4. 计算这个采样动作在当前分布下的对数概率
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value.item()


# 定义PPO算法
class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=3e-4,
                 gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
                 K_epochs=10, entropy_coef=0.01, value_loss_coef=0.5):

        # 计算gae
        self.gamma = gamma  # 0.99 折扣因子
        self.gae_lambda = gae_lambda  # 0.95 GAE系数，平衡偏差和方差

        self.clip_epsilon = clip_epsilon  # 0.2 PPO截断系数，(1-eps, 1+eps)
        self.K_epochs = K_epochs  # 每批数据更新次数 10，同一个数据迭代多次。
        self.entropy_coef = entropy_coef  # 熵正则化系数 添加随机项到总损失
        self.value_loss_coef = value_loss_coef  # 价值损失系数

        # 初始化Actor-Critic网络
        self.model = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def compute_gae(self, rewards, values, dones, next_value):
        """计算广义优势估计(GAE)、或者折扣回报也行
        GAE：它的核心逻辑是 “从后往前累积 TD 误差”，TD误差的加权和平衡偏差与方差。
        input:
            rewards: 时间步t到T-1的即时奖励列表（[r_t, r_{t+1}, ..., r_{T-1}]）
            values：价值网络对状态s_t到s_{T-1}的价值估计（[V(s_t), V(s_{t+1}), ..., V(s_{T-1})]）
            dones：标记每个状态是否为终止状态（[done_t, done_{t+1}, ..., done_{T-1}]，list. 终止时为 1，否则为 0）

            next_value：最后一个状态s_T的价值估计（V(s_T)， float32. 若s_T是终止状态则为 0）
        returns:
            advantages：每个时间步的优势估计（[A_t, A_{t+1}, ..., A_{T-1}]）
            returns：每个时间步的回报估计（[R_t, R_{t+1}, ..., R_{T-1}]，即Q(s_t,a_t)的估计）
        """
        advantages = []
        returns = []

        advantage = 0  # 临时变量，用于累积优势（从后往前算）
        # 从最后一步向前计算
        for t in reversed(range(len(rewards))):
            # 1. 计算TD误差（Temporal Difference Error）
            # 公式：delta_t = r_t + γ * V(s_{t+1}) * (1-done_t) - V(s_t)
            # 这是 “一步 TD 误差”，衡量 “当前价值估计V(s_t)” 与 “基于即时奖励和下一状态价值的估计” 之间的差距。
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]  # TD error
            next_value = values[t]  # 更新"下一个状态的价值"（因为是反向迭代，下一轮要处理t-1，此时t的价值就是t-1的"下一个价值"）

            # 2. 累积优势（GAE的核心）
            # 公式：A_t = delta_t + γ*λ * A_{t+1} * (1-done_t)
            advantage = delta + self.gamma * self.gae_lambda * advantage * (1 - dones[t])

            # 3. 回报 = 优势 + 价值（因为优势的定义A_t = Q_t - V(s_t)，而Q_t是理论值）
            return_val = advantage + values[t]

            # 4. 将当前优势和回报插入列表开头（保持时间顺序）
            advantages.insert(0, advantage)
            returns.insert(0, return_val)

        # 标准化优势，提高训练稳定性（可选但重要，避免不同量级的优势干扰训练）
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

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
        for _ in range(self.K_epochs):
            # 计算当前策略的logits和价值
            logits, values = self.model(states)  # states(18,4) logits(18,2) values(18,1)
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

        # 存储同属于一个episode的一个batch的数据
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        score_episode = 0

        for t in range(max_timesteps):  # 1000次，可能次episode没有结束，但是对于CartPole游戏最大是500次就会结束.
            # 选择动作，此动作的log_prob，此状态价值
            state_tensor = torch.tensor(state, dtype=torch.float32)  # tensor(4,)
            action, log_prob, value = ppo.model.get_action(state_tensor)  # actor网络给出动作、及其对应的概率、此状态价值

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
            score_episode += reward

            # 如果达到batch_size或者episode结束，更新策略
            if len(states) >= batch_size or done:
                # 获取最后一个状态的价值
                with torch.no_grad():
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
                    _, next_value = ppo.model(next_state_tensor)  # 评价网络，给出评价值-估计此状态价值
                    next_value = next_value.item()

                # 计算GAE和回报：计算同一个episode中不同时刻的gae和回报值
                advantages, returns = ppo.compute_gae(rewards, values, dones, next_value)

                # 更新PPO：一个episode中不同时刻的状态、动作、此动作概率、gae和回报值
                ppo.update(states, actions, log_probs, advantages, returns)

                # 重置存储的batch数据
                states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

            if done:
                break

        # 记录奖励
        scores.append(score_episode)
        scores_window.append(score_episode)

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
