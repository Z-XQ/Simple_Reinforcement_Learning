import math
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import gymnasium as gym
from sympy.physics.units import action
from tqdm import tqdm

"""
(1) 两个网络，都是输入s，输出q值，计算两者q差值，q=main_net(s), q = r + max(target_net(s))
(2) ReplayMemory, batch data
(3) on-policy
"""
class DQN(nn.Module):
    """"""
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.layer(x)


Transition = namedtuple("transition", field_names="state, action, reward, next_state")

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], capacity)

    def push(self, *args):
        transition = Transition(*args)
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class CartPoleTrainer(object):
    def __init__(self):
        # env
        self.env = gym.make("CartPole-v1")

        # model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.policy_net = DQN(state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 超参数
        self.step_done = 0
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 10000

        self.BATCH_SIZE = 128
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=1e-4, amsgrad=True)
        self.loss_function = torch.nn.SmoothL1Loss()
        self.memory = ReplayMemory(100000)
        self.UPDATE_TARGET_RATE = 10
        self.GAMMA = 0.99

    def train(self):
        # 记录batch data, 迭代模型
        for epoch in tqdm(range(3000)):
            state, _ = self.env.reset()
            over = False
            while not over:
                # 选取动作 eps-greedy
                action = self.get_action_by_eps_greedy(state)
                # 交互
                next_state, reward, terminate, truncated, info = self.env.step(action)
                over = terminate or truncated
                # 缓存
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,4)
                action_tensor = torch.tensor([action], dtype=torch.long, device=self.device).view(1, 1)
                reward_tensor = torch.tensor([reward], dtype=torch.long, device=self.device)
                if not over:
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,4)
                else:
                    next_state_tensor = None
                self.memory.push(state_tensor, action_tensor, reward_tensor, next_state_tensor)

                # 计算loss，更新模型
                self.update_policy_net()
                # next
                state = next_state

            # 定期更新target net
            if epoch % self.UPDATE_TARGET_RATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # 打印epoch信息
            # 打印训练过程中的一些信息
            if epoch % 100 == 0:
                # val
                val_result = sum([self.val_episode() for i in range(10)]) / 10
                print("average return on val: {}".format(val_result))

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

    def get_action_by_eps_greedy(self, state):
        """大概率选择q值最大的action，小概率随机一个动作"""
        # 计算eps
        eps = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1 * self.step_done / self.EPS_DECAY)
        self.step_done += 1
        p = random.random()
        if p > eps:
            with torch.no_grad():
                s_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,4)
                q_val = self.policy_net(s_tensor)  # (1,2)
                action = torch.argmax(q_val, dim=1).item()
        else:
            action = random.choices(range(self.action_dim), k=1)[0]

        return action

    def update_policy_net(self):
        """计算loss: q_val = main_net(s), target_q = r + γ*max(target_net(s))"""
        # 获取batch data
        if len(self.memory) < self.BATCH_SIZE:
            return

        batch_transition = self.memory.sample(self.BATCH_SIZE)  # list of t. [t1, t2, t3]
        # [t1, t2, t3] -> Transition(state=[s1,s2,s3], action=[a1,a2,a3], [r1,r2,r3], [s'1,s'2,s'3])
        transition = Transition(*zip(*batch_transition))
        state_batch = torch.cat(transition.state)  # list of s -> (128,4)
        action_batch = torch.cat(transition.action)  # list of s -> (128,1)
        reward_batch = torch.cat(transition.reward)  # list of s -> (128,)

        # 计算主网络q值: q=net(s)
        q_val = self.policy_net(state_batch)  # (128,4) -> (128,2)
        main_q_val = torch.gather(q_val, dim=1, index=action_batch)  # (128,1)

        # 计算目标网络q值: r + net(s')
        # s'如果为空，则q值为0，所以需要找到不为空的位置，计算q值
        not_none_list = [s is not None for s in transition.next_state]
        not_none_mask = torch.tensor(not_none_list, dtype=torch.bool, device=self.device)  # (128,)
        not_none_next_state_list = [s for s in transition.next_state if s is not None]
        not_none_next_state_tensor = torch.cat(not_none_next_state_list)  # (122,4)

        target_q_val = torch.zeros(size=(self.BATCH_SIZE, ), dtype=torch.float32, device=self.device)  # (128,)
        with torch.no_grad():
            # max(net(s))
            q_val = self.target_net(not_none_next_state_tensor)  # (122,2)
            q_val = torch.max(q_val, dim=1).values  # (122,)
            target_q_val[not_none_mask] = q_val
        target_q_val = reward_batch + self.GAMMA * target_q_val  # (128,)

        # 计算loss
        loss = self.loss_function(main_q_val, target_q_val.unsqueeze(1))

        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return



if __name__ == '__main__':
    trainer = CartPoleTrainer()
    trainer.train()
