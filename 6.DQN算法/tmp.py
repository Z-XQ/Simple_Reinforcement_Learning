import math
import os.path
import random
from collections import namedtuple, deque

import gymnasium as gym
from fontTools.misc.bezierTools import epsilon
from tqdm import tqdm
import torch
import torch.nn as nn


Transition = namedtuple(typename="Transition", field_names="state, action, reward, next_state")


class DQN(nn.Module):
    """q值近似函数"""
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_observations, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
            # nn.Softmax()  # 因为输出的是q值，不是概率值
        )

    def forward(self, x):
        return self.layer(x)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):  # 传入多个参数
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class CartPoleTrainer(object):
    def __init__(self, model_path="models/dqn0801.pth"):
        # 1. 检查gpu情况
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using gpu device name: {torch.cuda.get_device_name(0)} with {torch.cuda.device_count()} GPUs")
        else:
            print("Using cpu")

        # 2. env
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")

        # 3. 初始化双网络模型：主网络和目标网络
        self.n_observations = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.policy_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # 初始参数同步
        self.target_net.eval()  # 只是训练主网络policy_net

        # 4. 设置优化器与经验回放缓冲区
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=1e-4, amsgrad=True)
        self.memory = ReplayMemory(1000)

        # 5. 超参数
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99  # 折扣因子（未来奖励的衰减系数）
        self.TARGET_UPDATE = 10
        # eps-greedy
        self.step_done = 0
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000

        self.model_path = model_path
        save_dir = os.path.dirname(model_path)
        os.makedirs(save_dir, exist_ok=True)

    def select_action_eps_greedy(self, state):
        """eps-greedy策略获取action：eps公式选择action
        state: array. (4,)
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).reshape(1, 4)
        p = random.random()  # 概率
        self.step_done += 1
        eps = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1*self.step_done / self.EPS_DECAY)
        if p > eps:  # 选择最大q值对应的action
            with torch.no_grad():
                q_val = self.policy_net(state_tensor)
                action = torch.argmax(q_val, dim=1).item()  # 策略网络输出Q值，取最大值对应的动作索引
        else:  # 随机等概率选择一个action
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
        batch = Transition(
            *zip(*transitions))  # zip(*transitions) 等价于 zip(t1, t2): [(s1, s2), (a1, a2),(ns1, ns2),(r1, r2) ]
        # 拼接批量数据（状态、动作、奖励）
        state_batch = torch.cat(batch.state)  # list.(1,4) -> (128,4)
        action_batch = torch.cat(batch.action)  # list. (1,) -> (128,)
        reward_batch = torch.cat(batch.reward)  # list. (1,) -> (128,)

        # 2. 计算预测Q值=net(s)：当前状态下，实际选择的动作对应的Q值 s(128,4)->q(128,2)->q(128,1) gather在指定维度 dim 上，根据 index 中的索引值，提取对应位置的元素。
        action_batch = action_batch.unsqueeze(1)
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

        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)  # 梯度裁剪（防止梯度爆炸）
        self.optimizer.step()  # 更新策略网络参数

    def train(self):
        """
        on-policy. policy-net生成样本数据，缓存到队列中，并取batch样本进行迭代。
        policy-net计算的q值，和target-net计算的q值，最小化两者差值
        Returns:
        """
        # 生成one sample (s,a,r,s')，缓存起来，计算梯度，进行更新
        max_rewards = 0
        for i_episode in range(1500):
            # 1. 生成one sample (s,a,r,s')
            state, _ = self.env.reset()
            over = False
            while not over:
                action = self.select_action_eps_greedy(state)
                next_state, reward, terminate, truncated, info = self.env.step(action)
                over = terminate or truncated

                # 2. 缓存起来
                # array(4,) -> (1,4)
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).reshape(1, 4)
                # int. -> (1,)
                action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
                # float32 -> (1,)
                reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
                # array(4,) -> (1,4)
                if not over:
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).reshape(1, 4)
                else:
                    next_state_tensor = None
                # (s,a,r,s')
                self.memory.push(state_tensor, action_tensor, reward_tensor, next_state_tensor)

                # 3. 计算梯度，进行实时更新主网络，定期更新目标网络
                self.optimize_policy_model()

            # 打印训练过程中的一些信息
            if i_episode % 100 == 0:
                # val
                val_result = sum([self.val_episode() for i in range(10)]) / 10
                print("average return on val: {}".format(val_result))
                if val_result > max_rewards:
                    max_reward = val_result
                    torch.save(self.policy_net.state_dict(), self.model_path)

            # 4. 定期目标网络异步更新
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                # print(f"Target network updated at episode {i_episode}")
        print('训练完成')
        self.env.close()

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