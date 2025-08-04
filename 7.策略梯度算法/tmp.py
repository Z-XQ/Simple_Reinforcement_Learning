import os.path
import random

import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm


# p(s) = net(s)
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)  # 输出action概率
        )

    def forward(self, x):
        return self.layer(x)


class CartPoleTrainer(object):
    def __init__(self, render_mode="rgb_array", model_path="models/123.pth"):
        self.env = gym.make("CartPole-v1", max_episode_steps=500, render_mode=render_mode)
        self.policy_net = PolicyNet()

        self.model_path = model_path
        save_dir = os.path.dirname(model_path)
        os.makedirs(save_dir, exist_ok=True)

    def get_action_by_p(self, state):
        """

        Args:
            state: array. (4,)

        Returns: int. 0 / 1

        """
        state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, 4)
        prob = self.policy_net(state_tensor)  # (1,2)
        action = random.choices(range(2), weights=prob[0].tolist(), k=1)[0]
        return action

    def get_action_by_max(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, 4)
        prob = self.policy_net(state_tensor)  # (1,2)
        action = torch.argmax(prob, dim=1).item()  # (1,1)
        return action

    def get_episode(self):
        states, actions, rewards = [], [], []
        state, _  = self.env.reset()
        over = False
        while not over:
            action = self.get_action_by_p(state)
            next_state, reward, terminate, truncated, info = self.env.step(action)
            # 记录
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            # next
            over = terminate or truncated
            state = next_state

        return states, actions, rewards

    def train(self):
        optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        val_g_max = 0
        for epoch in tqdm(range(1000)):
            # 清空梯度
            optimizer.zero_grad()

            # 计算梯度
            # 获取数据
            states, actions, rewards = self.get_episode()
            reward_sum = 0
            for i in reversed(range(len(states))):
                # 最大化 lnp * q，最大化某个s-a配对下的q值和采取此动作的概率，也就是最小化-lnp * q
                # 先求q
                reward_sum = rewards[i] + 0.99*reward_sum
                # 再求p
                state_tensor = torch.tensor(states[i], dtype=torch.float32).reshape(1, 4)
                prob = self.policy_net(state_tensor)
                cur_p = prob[0, actions[i]]
                # loss
                loss = -cur_p.log() * reward_sum
                loss.backward(retain_graph=True)
            # 利用梯度更新参数
            optimizer.step()

            # val
            if epoch % 100 == 0:
                val_g_res = sum([self.val_episode() for i in range(10)]) / 10
                print("val rewards on val: {}".format(val_g_res))
                if val_g_res > val_g_max:
                    val_g_max = val_g_res
                    torch.save(self.policy_net.state_dict(), self.model_path)

    def val_episode(self):
        val_res = 0
        state, _ = self.env.reset()
        over = False
        while not over:
            action = self.get_action_by_max(state)
            next_s, reward, terminate, truncated, info = self.env.step(action)
            val_res += reward
            over = terminate or truncated
            state = next_s

        return val_res


class CartPoleTester:
    def __init__(self, model_path="1.pth", render_mode="human"):
        self.model_path = model_path
        self.model = PolicyNet()
        self._load_model()

        self.env = gym.make("CartPole-v1", max_episode_steps=500, render_mode=render_mode)

    def _load_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path))
        except Exception as e:
            print("加载模型失败：", e)
            raise

    def get_action_by_max_p(self, state):
        state = torch.FloatTensor(state).reshape(1, 4)
        with torch.no_grad():  # 测试时不需要计算梯度
            prob = self.model(state)
        action = torch.argmax(prob).item()  # 测试时使用确定性策略（选择概率最高的动作）
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
                action = self.get_action_by_max_p(state)  # 最优策略
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

    # 使用训练好的模型进行测试
    tester = CartPoleTester(model_path="models/123.pth", render_mode='human')  # 'human'模式用于可视化
    tester.test(episode_num=5)