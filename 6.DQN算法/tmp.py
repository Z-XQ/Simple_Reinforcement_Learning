import math
from collections import deque, namedtuple

import torch
import torch.nn as nn
import random
import gymnasium as gym
from tqdm import tqdm

"""
DQN: deep q network
(1) 两个网络：q = policy_net(s), target_q = r + max(target_net(s')), loss = mse(target_q - q)
(2) replayMemory, batch_data train, batch个时刻的数据;
(3) on-policy
"""

Transition = namedtuple("transition", field_names="state, action, reward, next_state")

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], capacity)

    def push(self, *args):
        t = Transition(*args)
        self.memory.append(t)

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """q = net(s)"""
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        """
        q = net(s)
        Args:
            state: tensor. (1, 4)

        Returns:

        """
        return self.layer(state)


class CartPoleTrainer(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # envs
        self.env = gym.make("CartPole-v1")

        # model
        state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.policy_net = DQN(state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(state_dim, self.action_dim).to(self.device)
        self.target_net.eval()

        # 超参数
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 10000
        self.step_done = 0

        self.memory = ReplayMemory(100000)
        self.TARGET_UPDATE_RATE = 10  # 目标网络更新频率（每10个episode更新一次）

    def get_action_by_eps_greedy(self, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, 4)
        eps = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1.0 * self.step_done / self.EPS_DECAY)
        self.step_done += 1
        p = random.random()
        if p > eps:
            with torch.no_grad():
                q = self.policy_net(state_t)  # (1, 2)
                action = torch.argmax(q, dim=1).item()
        else:
            action = random.choices(range(self.action_dim), k=1)[0]
        return action

    def train(self):
        # 获取batch data
        for epoch in tqdm(range(1500)):
            state, _ = self.env.reset()
            over = False
            while not over:
                # 获取动作
                action = self.get_action_by_eps_greedy(state)
                # 交互
                next_state, reward, terminate, truncated, info = self.env.step(action)
                over = terminate or truncated

                # 存储
                state_t = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, 4)
                action_t = torch.tensor([action], dtype=torch.float32, device=self.device).view(1, 1)
                reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)
                if not over:
                    next_state_t = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, 4)
                else:
                    next_state_t = None

                self.memory.push(state_t, action_t, reward_t, next_state_t)

                # 迭代 batch 个数据
                self.update_policy_model()

                # next
                state = next_state

            # eval
            if epoch % 100 == 0:
                pass

            # update target net
            if epoch % self.TARGET_UPDATE_RATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        # 计算q

    def update_policy_model(self):
        """
        q = net(s), r+max(net(s'))
        Returns:

        """
        # 获取batch data
        if len(self.memory) < self.BATCH_SIZE:
            return

        batch = self.memory.sample(self.BATCH_SIZE)  # list of t. [t1, t2, t3]
        # [t1, t2, t3] -> {s=[s1,s2,s3], a=[a1,a2,a3], r=[r1,r2,r3], s'=[s'1,s'2,s'3]}
        t = Transition(*zip(*batch))
        state_batch = torch.cat(t.state)  # list of tensor. (1,4) -> (128,4)
        action_batch = torch.cat(t.action)  # list of tensor. (1,1) -> (128,1)
        reward_batch = torch.cat(t.reward)  # list of tensor. (1,) -> (128,)

        # q = policy_net(s)
        q_val = self.policy_net(state_batch)  # (128,4) -> (128,2)
        q_val = torch.gather(q_val, dim=1, index=action_batch)  # (128,1)

        # target_q = r + max(target_net(s'))
        # 最后一个state没有q值，只计算batch内不是最后state的q值
        not_none_list = [s is not None for s in t.next_state]
        not_none_mask = torch.tensor(not_none_list, dtype=torch.bool, device=self.device)  # (128,1)
        not_none_next_states = [s for s in t.next_state if s is not None]
        not_none_next_state_batch = torch.cat(not_none_next_states)  # (122,4)

        target_q = torch.zeros(size=(self.BATCH_SIZE, ), dtype=torch.float32, device=self.device)  # (128,)
        with torch.no_grad():
            target_q_val = self.target_net(not_none_next_state_batch)  # (122,2)
            target_q_val = torch.max(target_q_val, dim=1).values  # (122,)
            target_q[not_none_mask] = target_q_val
        # reward_batch(128,) target_q(128,)
        target_q = reward_batch + self.GAMMA*target_q

        # loss
        loss = torch.nn.SmoothL1Loss(target_q, q_val)

        # update