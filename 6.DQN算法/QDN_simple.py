import os.path

import gymnasium as gym
import math
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

"""与之前功能一致，用namedtuple定义强化学习中一次完整交互的经验（状态、动作、下一状态、奖励），
支持通过字段名访问（如transition.state），提升可读性。"""
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


# 类似于dataset
class ReplayMemory(object):
    """
    作用：存储历史交互经验，训练时随机采样批量数据，避免连续样本的相关性过高导致的训练不稳定。
    类比：相当于一个 "经验数据库"，push用于写入新经验，sample用于随机读取批量数据。
    队列存储全部的缓存，该队列最长是capacity个sample，超过则先进先出掉。
    通过push和sample入队和批量采样。
    """
    def __init__(self, capacity):
        # 预先申请空间。第一个参数 []：初始化一个空队列（也可以传入初始元素列表，如 deque([t1,t2,t3], ...)）。
        self.memory = deque([], maxlen=capacity)  # capacity个样本，一个样本就是一个Transition

    def push(self, *args):
        """保存一个转换"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """随机获取batch data"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)  # 返回当前存储的经验数量


class DQN(nn.Module):
    """DQN 的核心是通过神经网络拟合 Q 值函数（输入状态，输出每个动作的 Q 值，即 "状态 - 动作价值"）。"""
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class CartPoleTrainer():
    def __init__(self, model_path="models/dqn.pth"):
        # 1. 检查是否使用 GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)} with {torch.cuda.device_count()} GPUs")
        else:
            print("Using CPU")

        # 2. 设置环境
        self.env = gym.make('CartPole-v1')

        # 3. 双网络初始化（策略网络+目标网络）
        n_observations = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.policy_net = DQN(n_observations, self.n_actions).to(self.device)  # “实时”更新的策略网络
        self.target_net = DQN(n_observations, self.n_actions).to(self.device)  # “定期”更新的目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())  # 初始参数同步
        self.target_net.eval()  # 目标网络设为评估模式（不计算梯度）

        # 4. 设置优化器(amsgrad稳定学习率，避免过早衰减)与经验回放缓冲区
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=1e-4, amsgrad=True)
        self.memory = ReplayMemory(10000)

        # 5. 超参数定义
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99        # 折扣因子（未来奖励的衰减系数）
        self.TARGET_UPDATE = 10  # 目标网络更新频率（每10个episode更新一次）

        # eps-greedy
        self.steps_done = 0  # 累计步数（用于ε-贪婪策略的衰减）
        self.EPS_START = 0.9     # ε-贪婪策略初始探索率
        self.EPS_END = 0.05      # ε-贪婪策略最终探索率
        self.EPS_DECAY = 1000    # ε衰减速度

        self.model_path = model_path
        save_dir = os.path.dirname(model_path)
        os.makedirs(save_dir, exist_ok=True)

    def select_action_eps_greedy(self, state):
        """
        核心逻辑：实现ε-贪婪策略，平衡 "探索"（随机尝试新动作）和 "利用"（选择当前最优动作）：
            初始阶段 ε=0.9（高探索率），优先尝试新动作；
            随训练步数增加，ε逐渐衰减到 0.05（低探索率），逐渐依赖策略网络决策。
        Args:
        input:
            state: array. (4,)
        Returns:
            int. action idx.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
            0)  # array(4,) -> tensor(1,4)

        # 计算当前ε值（随步数指数衰减）
        eps = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)

        self.steps_done += 1
        p = random.random()
        if p > eps:  # 利用：选择当前Q值最大的动作
            with torch.no_grad():  # 不计算梯度（节省资源）
                q_val = self.policy_net(state_tensor)
                action = torch.argmax(q_val, dim=1).item()  # 策略网络输出Q值，取最大值对应的动作索引
        else:  # 探索：随机选择动作，所以随机某个概率是eps*1/n
            action = random.choices(range(self.n_actions), k=1)[0]

        return action

    def optimize_policy_model(self):
        """
        optimize_policy_model函数实现 DQN 的核心训练步骤，通过最小化 "预测 Q 值" 与 "目标 Q 值" 的差距更新策略网络。
        核心逻辑：通过贝尔曼方程将 Q 值学习转化为监督学习问题，目标 Q 值 = 即时奖励 + γ× 下一状态最大 Q 值（γ 为折扣因子）。
        Returns:

        """
        if len(self.memory) < self.BATCH_SIZE:  # 经验不足时不训练
            return

        # 1. 随机采样一批经验, 转换为batch批量数据（按字段分组）
        transitions = self.memory.sample(self.BATCH_SIZE)  # list of Transition
        batch = Transition(*zip(*transitions))  # zip(*transitions) 等价于 zip(t1, t2): [(s1, s2), (a1, a2),(ns1, ns2),(r1, r2) ]
        # 拼接批量数据（状态、动作、奖励）
        state_batch = torch.cat(batch.state)  # list.(1,4) -> (128,4)
        action_batch = torch.cat(batch.action)  # list. (1,1) -> (128,1)
        reward_batch = torch.cat(batch.reward)  # list. (1,) -> (128,)

        # 2. 计算预测Q值=net(s)：当前状态下，实际选择的动作对应的Q值 s(128,4)->q(128,2)->q(128,1) gather在指定维度 dim 上，根据 index 中的索引值，提取对应位置的元素。
        state_action_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)

        # 3. 计算目标Q值：基于目标网络的下一状态最大Q值（贝尔曼方程）: r + max(net(s'), dim=1)
        # 处理非终结状态（下一个状态存在的情况）即(s,a,r,s'=None)不参与训练。 tensor.shape=(128,)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])  # (122,4)
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)  # 终结状态的目标 Q 值为0：从该状态出发，不会再有任何后续动作或奖励
        with torch.no_grad():
            # next_state_values[non_final_mask].shape=(122,) 对应 non_final_next_states.shape=(122,4), 只计算非终结状态的目标Q值
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(dim=1)[0]
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch  # 贝尔曼更新

        # 4. 计算损失并优化
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()   # 清空梯度
        loss.backward()  # 反向传播计算梯度
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)   # 梯度裁剪（防止梯度爆炸）
        self.optimizer.step()   # 更新策略网络参数

    def train(self):
        # 训练过程
        max_reward = 0
        for i_episode in tqdm(range(1500)):  # 1500个episode
            # 单个episode过程：生成one sample (s,a,r,s')，缓存起来，计算梯度，进行实时更新主网络，定期更新目标网络
            state, _ = self.env.reset()
            over = False
            while not over:
                # 1. 生成one sample (s,a,r,s')
                # 主网络选取策略，进行交互，产生一个Transition训练样本
                # on-policy策略，采用eps_greedy生成多样性数据
                action = self.select_action_eps_greedy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                over = terminated or truncated

                # 2. 缓存起来(s,a,r,s')
                # array(4,) -> tensor(1,4)
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                # int -> (1,)
                action_tensor = torch.tensor([action], device=self.device, dtype=torch.long).view(1, 1)
                # float32 -> (1,)
                reward_tensor = torch.tensor([reward], device=self.device)
                # array(4,) -> tensor(1,4)
                if not over:
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    next_state_tensor = None
                self.memory.push(state_tensor, action_tensor, reward_tensor, next_state_tensor)

                # next iteration
                state = next_state

                # 3. 计算梯度，进行实时更新主网络
                self.optimize_policy_model()

            # 打印训练过程中的一些信息
            if i_episode % 100 == 0:
                # val
                val_result = sum([self.val_episode() for i in range(10)]) / 10
                print("average return on val: {}".format(val_result))
                if val_result > max_reward:
                    max_reward = val_result
                    torch.save(self.policy_net.state_dict(), self.model_path)

            # 4. 定期目标网络异步更新
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                # print(f"Target network updated at episode {i_episode}")

        print('训练完成')
        self.env.close()

    def get_action_by_max_q(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).reshape(1, 4)
        with torch.no_grad():  # 测试时不需要计算梯度
            q_val = self.policy_net(state)
        action = torch.argmax(q_val).item()  # 选择q值最高的动作
        return action

    def val_episode(self):
        # 重置环境，随机一个初始状态
        state, info = self.env.reset()
        reward_sum = 0  # 计算总分，走一步没结束则加一分
        over = False
        while not over:
            # 输出策略
            action = self.get_action_by_max_q(state)
            # 交互
            state, reward, terminated, truncated, info = self.env.step(action)
            reward_sum += reward
            # next
            over = terminated or truncated

        return reward_sum


class CartPoleTester:
    def __init__(self, model_path="models/dqn.pth", render_mode="human"):
        # 1. 检查是否使用 GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)} with {torch.cuda.device_count()} GPUs")
        else:
            print("Using CPU")

        self.model_path = model_path
        self.q_net = DQN(4, 2).to(self.device)
        self.q_net.eval()
        self._load_model()

        self.env = gym.make("CartPole-v1", max_episode_steps=500, render_mode=render_mode)

    def _load_model(self):
        try:
            self.q_net.load_state_dict(torch.load(self.model_path))
        except Exception as e:
            print("加载模型失败：", e)
            raise

    def get_action_by_max_q(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).reshape(1, 4)
        with torch.no_grad():  # 测试时不需要计算梯度
            q_val = self.q_net(state)
        action = torch.argmax(q_val).item()  # 选择q值最高的动作
        return action

    def test(self, episode_num=5):
        print(f"\n开始测试，共运行 {episode_num} 局...")

        total_reward = 0  # 多局游戏的平均奖励
        for i in range(episode_num):
            state, _ = self.env.reset()
            reward_sum = 0  # 一局游戏奖励
            over = False
            while not over:
                # 采取最优策略
                action = self.get_action_by_max_q(state)  # 最优策略
                # action = self.env.action_space.sample()  # 随机采取某个动作

                # 交互
                next_state, reward, terminate, truncated, info = self.env.step(action)
                reward_sum += reward

                # 下一个次迭代
                state = next_state
                over = terminate or truncated

            print(f"第 {i + 1} 局结束，奖励: {reward_sum}")
            total_reward += reward_sum

        avg_reward = total_reward / episode_num
        print(f"\n测试完成，平均奖励: {avg_reward:.2f}")
        return avg_reward

if __name__ == '__main__':
    trainer = CartPoleTrainer()
    trainer.train()

    tester = CartPoleTester(model_path="models/dqn.pth")
    tester.test()