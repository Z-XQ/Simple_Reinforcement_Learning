import math
import random
from collections import deque, namedtuple

import torch
from torch import nn, no_grad

import gymnasium as gym
from tqdm import tqdm

Transition = namedtuple(typename="transition", field_names="state, action, reward, next_state")

class DQN(nn.Module):
    """三层网络，q=net(s)"""
    def __init__(self, n_s, n_a):
        super(DQN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_s, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_a)
        )

    def forward(self, x):
        return self.layer(x)


class ReplayMemory(object):
    """{(s, a, r, s')}"""
    def __init__(self, n_size):
        self.memory = deque([], maxlen=n_size)

    def push(self, *args):
        """input: s, a, r, s' """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class CartPoleTrainer(object):
    """两个网络：训练主网络实时更新，计算一个q值，不训练目标网络定期更新，也计算一个q值，迭代缩小这两个q值的差值"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. env
        self.env = gym.make("CartPole-v1")

        # 2. net
        n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.policy_net = DQN(n_states, self.n_actions).to(self.device)
        self.target_net = DQN(n_states, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 3. 参数  启用amsgrad=True会让优化过程更稳定，尤其在训练周期较长或优化目标较复杂的场景下
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), 1e-4, amsgrad=True)
        self.loss_function = nn.SmoothL1Loss()
        self.memery = ReplayMemory(10000)

        self.BATCH_SIZE = 128
        self.GAMMA = 0.99  # 折扣因子
        self.TARGET_UPDATE_RATE = 10

        self.step_done = 0
        self.EPS_START = 0.90
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000

    def get_action_by_eps_greedy(self, state):
        """

        Args:
            state: array. (4, )

        Returns: int. 0/1

        """
        s_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 4)
        eps = self.EPS_END + (self.EPS_START-self.EPS_END) * math.exp(-1 * self.step_done / self.EPS_DECAY)
        self.step_done += 1

        p = random.random()
        if p > eps:
            with torch.no_grad():  # 最大q值的动作
                q_val = self.policy_net(s_tensor)
                action = torch.argmax(q_val, dim=1).item()
        else:
            action = random.choices(range(self.n_actions), k=1)[0]
        return action

    def optimize_policy_model(self):
        """
        计算两个网络的q值，最小化此差值
        Returns:

        """
        if len(self.memery) < self.BATCH_SIZE:
            return

        # batch data
        batch_data = self.memery.sample(self.BATCH_SIZE)  # [t1, t2, t3]
        #  [t1, t2, t3] ->Transition { s=[s1,s2,s3], a=[a1,a2,a3], r=[r1,r2,r3], s'=[s'1,s'2,s'3]}
        transition = Transition(*zip(*batch_data))
        s_batch = torch.cat(transition.state)  # list of tensor: (1,4) -> (128,4)
        a_batch = torch.cat(transition.action)  # (1,1) -> (128,1)
        r_batch = torch.cat(transition.reward)  # (1,) -> (128,)

        # 主网络q值
        main_q_val = self.policy_net(s_batch)  # (128,4) -> (128,2)
        cur_q_a_val = torch.gather(main_q_val, dim=1, index=a_batch)  # (128,1)

        # 目标网络q值
        # 但是next_state存在为None情况，此时不存在q值设为0，所以只计算不为None位置的q值
        not_none_list = [s is not None for s in transition.next_state]
        not_none_mask = torch.tensor(not_none_list, dtype=torch.bool, device=self.device)  # (128,)
        next_state_list = [s for s in transition.next_state if s is not None]
        next_state_batch = torch.cat(next_state_list)  # (122,4)

        target_q_val = torch.zeros(size=(self.BATCH_SIZE, ), dtype=torch.float32, device=self.device)  # (128,)
        with torch.no_grad():
            q_val = self.target_net(next_state_batch)  # (122,4) -> (122,2)
            target_q_val[not_none_mask] = torch.max(q_val, dim=1).values  # (122,) -> (128,)
        target_q_val = r_batch + self.GAMMA*target_q_val

        loss = self.loss_function(cur_q_a_val, target_q_val.unsqueeze(1))
        print(loss)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self):
        max_val_rewards = 0
        for epoch in tqdm(range(1500)):  # 1500 episodes
            over = False
            state, _ = self.env.reset()
            while not over:
                # 选出动作
                action = self.get_action_by_eps_greedy(state)

                # 采取动作
                next_state, reward, terminate, truncated, info = self.env.step(action)
                over = terminate or truncated

                # 缓存
                state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                action_t = torch.tensor([action], dtype=torch.long, device=self.device).view(1, 1)
                reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)
                if not over:
                    next_s_t = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    next_s_t = None
                self.memery.push(state_t, action_t, reward_t, next_s_t)

                # 更新迭代
                self.optimize_policy_model()
            # 打印训练过程中的一些信息
            if epoch % 100 == 0:
                # val
                val_result = sum([self.val_episode() for i in range(10)]) / 10
                print("average return on val: {}".format(val_result))
                if val_result >= max_val_rewards:
                    max_val_rewards = val_result
                    # torch.save(self.policy_net.state_dict(), self.model_path)

            # 4. 定期目标网络异步更新
            if epoch % self.TARGET_UPDATE_RATE == 0:
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
