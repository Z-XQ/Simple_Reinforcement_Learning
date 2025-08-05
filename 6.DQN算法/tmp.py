import math
import os
import random
from collections import namedtuple, deque

import gymnasium as gym
import torch
import torch.nn as nn
from tqdm import tqdm

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
        self.policy_net = DQN(n_states, self.n_actions).to(self.device)
        self.target_net = DQN(n_states, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 4. 设置优化器(amsgrad稳定学习率，避免过早衰减)与经验回放缓冲区
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=1e-4, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()
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
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
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

    def train(self):
        """dqn两个特点：双网络，重放"""
        max_reward = 0
        for epoch in tqdm(range(1500)):  # 1500个episode
            # 获取数据
            state, _ = self.env.reset()
            over = False
            while not over:
                # policy net获取动作
                action = self.select_action_by_eps_greedy(state)
                # 交互
                next_state, reward, terminate, truncated, info = self.env.step(action)
                over = terminate or truncated
                # 缓存数据 s,a,r,s'
                # (4, ) ->(1, 4)
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                # int -> (1, 1)
                action_tensor = torch.tensor([action], dtype=torch.long, device=self.device).reshape(1, 1)
                # float -> (1, )
                reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
                # (4, ) ->(1, 4)
                if not over:
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    next_state_tensor = None
                self.memory.push(state_tensor, action_tensor, reward_tensor, next_state_tensor)

                # 清空梯度 计算梯度 更新参数
                self.optimize_model()

                # next
                state = next_state

            # 打印训练过程中的一些信息
            if epoch % 100 == 0:
                # val
                val_result = sum([self.val_episode() for i in range(10)]) / 10
                print("average return on val: {}".format(val_result))
                if val_result > max_reward:
                    max_reward = val_result
                    torch.save(self.policy_net.state_dict(), self.model_path)

            # 4. 定期目标网络异步更新
            if epoch % self.TARGET_UPDATE_RATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                # print(f"Target network updated at episode {i_episode}")

        print('训练完成')
        self.env.close()


    def optimize_model(self):
        """
        主网络q=net(s), 目标网络q=r+max_q(s', a)
        Returns:
        """
        # 获取batch data
        if len(self.memory) < self.BATCH_SIZE:
            return
        batch_data = self.memory.sample(self.BATCH_SIZE)  # list.
        # zip(*batch_data) = zip(t1, t2, t3): [(s1,s2,s3), (a1,a2,a3), (r1,r2,r3), (s'1,s'2,s'3)]
        batch = Transition(*zip(*batch_data))
        state_t = torch.cat(batch.state, dim=0)  # tuple of tensor(1,4) -> (128, 4)
        action_t = torch.cat(batch.action, dim=0)  # tuple of tensor(1,1) -> (128, 1)
        reward_t = torch.cat(batch.reward, dim=0)  # tuple of tensor(1,) -> (128, )
        # next_s_t = torch.cat(batch.next_state, dim=0)  # 不能cat, 因为tuple of tensor(1,4) or None

        # 主网络q值
        main_q_val = self.policy_net(state_t)  # (128,4) -> (128,2)
        main_q_val = torch.gather(main_q_val, dim=1, index=action_t)  # (128,2)->(128,)

        # 目标网络q值: (128, ) + max(q)
        # max(q(s'))，但是如果s是最后的状态，则s'=None，不能输入到网络中计算q值
        # 找到next_s_t(128, 4)中等于None的所有位置
        not_none_mask = [s is not None for s in batch.next_state]
        not_none_mask = torch.tensor(not_none_mask, dtype=torch.bool, device=self.device)  # (128, )
        not_none_next_state = [s for s in batch.next_state if s is not None]  # list of (1, 4)
        not_none_next_state = torch.cat(not_none_next_state, dim=0)  #  (124, 4)

        next_max_q = torch.zeros(size=(self.BATCH_SIZE, ), dtype=torch.float32, device=self.device)   # (128, )
        # next_max_q.shape=(128, )
        next_max_q[not_none_mask] = torch.max(self.target_net(not_none_next_state), dim=1).values  # (128, )
        target_q = reward_t + self.GAMMA*next_max_q  # (128, )
        # 求loss
        loss = self.criterion(main_q_val, target_q.unsqueeze(1))

        # 梯度，更新参数
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

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

    def get_action_by_max_q(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).reshape(1, 4)
        with torch.no_grad():  # 测试时不需要计算梯度
            q_val = self.policy_net(state)
        action = torch.argmax(q_val).item()  # 选择q值最高的动作
        return action

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
    # trainer = CartPoleTrainer()
    # trainer.train()

    tester = CartPoleTester(model_path="models/123.pth")
    tester.test()