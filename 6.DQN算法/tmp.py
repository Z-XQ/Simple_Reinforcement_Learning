import math
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm

# DQN
"""
(1) DQN网络：q = net(s), s. array(1,4), q. (1,2)
(2) 主网络，目标网络，最小化这两者输出的q值
(3) Replay batch data
"""

class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_states, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.layer(x)


Transition = namedtuple(typename="transition", field_names="state, action, reward, next_state")


class ReplayMemory(object):
    def __init__(self, n_samples):
        self.memory = deque([], n_samples)

    def push(self, *args):
        transition = Transition(*args)
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class CartPoleTrainer(object):
    def __init__(self):
        self.env = gym.make("CartPole-v1", render_mode="human")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # two model
        n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.policy_net = DQN(n_states, self.n_actions).to(self.device)
        self.target_net = DQN(n_states, self.n_actions).to(self.device)
        self.target_net.eval()

        # 超参数
        self.BATCH_SIZE = 128
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=1e-4, amsgrad=True)
        self.loss_function = nn.SmoothL1Loss()
        self.memory = ReplayMemory(10000)
        self.GAMMA = 0.99
        self.UPDATE_TARGET_RATE = 10

        # eps-greedy
        self.step_done = 0
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000


    def get_action_by_eps_greedy(self, state):
        s_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        # eps 随着step逐渐减小
        eps = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1 * self.step_done / self.EPS_DECAY)

        p = random.random()
        if p > eps:
            with torch.no_grad():
                q_val = self.policy_net(s_tensor)
                action = torch.argmax(q_val, dim=1).item()
        else:
            action = random.choices(range(self.n_actions), k=1)[0]

        return action

    def optimize_policy_model(self):
        """
        获取batch data，分别计算两者的q值，最小化两者差值
        Returns:

        """
        if len(self.memory) < self.BATCH_SIZE:
            return

        # 获取batch data
        batch_transitions = self.memory.sample(self.BATCH_SIZE)  # list of transition. [t1, t2, t3]
        # [t1, t2, t3] -> {next_state=[s1,s2,s3], [a1,a2,a3], [r1,r2,r3], [s'1,s'2,s'3]}
        batch_data = Transition(*zip(*batch_transitions))
        state_batch = torch.cat(batch_data.state)  # list of tensor(1,4) -> (128,4)
        action_batch = torch.cat(batch_data.action)  # list of tensor(1,1) -> (128,1)
        reward_batch = torch.cat(batch_data.reward)  # list of tensor(1,) -> (128,)

        # 计算主网络q值=net(s)
        main_q_val = self.policy_net(state_batch)  # (128, 2)
        main_q_val = torch.gather(main_q_val, dim=1, index=action_batch)  # (128, 1)

        # 计算目标网络q值 = r + max(net(s'))
        # 如果s'为空，计算不了q值，s'为空的，q值设置为0
        not_none_list = [s is not None for s in batch_data.next_state]
        not_none_mask = torch.tensor(not_none_list, dtype=torch.bool, device=self.device)
        not_none_next_states = [s for s in batch_data.next_state if s is not None]
        not_none_next_states = torch.cat(not_none_next_states)  # (122,4)

        target_q_val = torch.zeros(size=(self.BATCH_SIZE, ), dtype=torch.float32, device=self.device)  # (128,)
        with torch.no_grad():
            q_val = self.target_net(not_none_next_states)  # (122,4) -> (122,2)
            max_q_val = torch.max(q_val, dim=1).values  # (122,)
            target_q_val[not_none_mask] = max_q_val  # (128,)
        target_q_val = reward_batch + self.GAMMA * target_q_val  # (128,)

        # 计算loss
        loss = self.loss_function(main_q_val, target_q_val.unsqueeze(1))

        # 清空梯度
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self):
        max_reward = 0
        for epoch in tqdm(range(1500)):
            over = False
            state, _ = self.env.reset()
            while not over:
                # 获取数据
                # 选取动作
                action = self.get_action_by_eps_greedy(state)
                # 交互
                next_state, reward, terminate, truncated, info = self.env.step(action)
                over = terminate or truncated

                # 缓存数据
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 4)
                action_tensor = torch.tensor([action], dtype=torch.long, device=self.device).view(1, 1)  # (1,1)
                reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)  # (1,)
                if not over:
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 4)
                else:
                    next_state_tensor = None
                self.memory.push(state_tensor, action_tensor, reward_tensor, next_state_tensor)

                self.optimize_policy_model()

                # 下一个迭代
                state = next_state

            # 更新目标模型
            if epoch % self.UPDATE_TARGET_RATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # 打印训练过程中的一些信息
            if epoch % 100 == 0:
                # val
                val_result = sum([self.val_episode() for i in range(10)]) / 10
                print("average return on val: {}".format(val_result))
                if val_result >= max_reward:
                    max_reward = val_result
                    # torch.save(self.policy_net.state_dict(), self.model_path)

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

if __name__ == '__main__':
    trainer = CartPoleTrainer()
    trainer.train()