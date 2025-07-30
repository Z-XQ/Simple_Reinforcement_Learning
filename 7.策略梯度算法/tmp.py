import os
import random

import gymnasium as gym
import torch
from torch import nn
from tqdm import tqdm


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layer(x)


class CartPoleTrainer:
    def __init__(self, render_mode="rgb_array", max_step=500, model_path="1.pth", learning_rate=1e-3):
        self.env = gym.make("CartPole-v1", max_episode_steps=max_step, render_mode=render_mode)
        self.model = SimpleModel()

        self.model_path = model_path
        self.learning_rate = learning_rate
        save_dir = os.path.dirname(model_path)
        os.makedirs(save_dir, exist_ok=True)

    def get_action_by_sample(self, state):
        """

        Args:
            state: float32. array. (4,)

        Returns: int. 0/1

        """
        state_tensor = torch.FloatTensor(state).reshape(1, 4)
        output_prob = self.model(state_tensor)  # (1,4) -> (1,2)

        # sample
        action = random.choices(range(2), weights=output_prob[0].tolist(), k=1)[0]
        return action

    def get_action_by_max_p(self, state):
        state_tensor = torch.FloatTensor(state).reshape(1, 4)
        output_prob = self.model(state_tensor)  # (1,4) -> (1,2)

        # sample
        action = torch.argmax(output_prob, dim=1).item()
        return action

    def get_episode(self):
        states = []
        actions = []
        rewards = []

        state, _ = self.env.reset()  # 初始化环境
        over = False
        while not over:
            # 根据策略模型，选取动作
            action = self.get_action_by_sample(state)
            # 与环境交互
            next_state, reward, terminal, truncated, info = self.env.step(action)
            # 缓存
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # 下一步
            state = next_state
            over = terminal or truncated
        return states, actions, rewards

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        max_reward_sum = 0

        for epoch in tqdm(range(1000)): # 1000条episode
            # 先获取一条episode数据
            states, actions, rewards = self.get_episode()

            # 计算q值，计算loss
            optimizer.zero_grad()  # 清空梯度

            reward_sum = 0  # 近似q值
            for i in reversed(range(len(states))):  # 更新模型参数，最大化采取states[i], actions[i]时的期望q值
                # 更新模型参数，最大化log(p) * q，可以最小化此loss函数。
                # 计算loss = -log(p) * p
                reward_sum = rewards[i] + 0.99*reward_sum

                # 根据策略模型，获取动作概率Π(a|s)
                state_tensor = torch.FloatTensor(states[i]).reshape(1, 4)
                prob = self.model(state_tensor)  # (1,2)

                # loss 更新模型参数，最大化log(p) * q，可以最小化此loss函数。
                loss = -prob[0, actions[i]].log() * reward_sum
                loss.backward(retain_graph=True)

            # 更新模型参数
            optimizer.step()

            # val
            if epoch % 100 == 0:
                val_result = sum([self.val_episode() for i in range(10)]) / 10
                print(f"Epoch: {epoch}, Test Reward: {val_result}")

                # 保存模型
                if reward_sum >= max_reward_sum:
                    max_reward_sum = reward_sum
                    torch.save(self.model.state_dict(), self.model_path)

    def val_episode(self):
        reward_sum = 0  # 走一步没有结束则得一分
        over = False
        state, _ = self.env.reset()
        while not over:
            # 根据策略模型，获取应该采取的动作
            action = self.get_action_by_max_p(state)
            # 采取动作，交互获得分数
            next_state, reward, terminate, truncated, info = self.env.step(action)
            reward_sum += reward
            # 下一步
            state = next_state
            over = terminate or truncated
        return reward_sum


# 加载模型，获取得分
class CartPoleTester:
    def __init__(self, model_path):
        self.model = SimpleModel()
        self.model_path = model_path
        self._load_weight()
        self.env = gym.make("CartPole-v1", max_episode_steps=500, render_mode="human")

    def _load_weight(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path))
        except Exception as e:
            print(e)
            raise

    def get_action_by_max_p(self, state):
        state_tensor = torch.FloatTensor(state).reshape(1, 4)
        prob = self.model(state_tensor)  # (1,2)
        action = torch.argmax(prob, dim=1).item()
        return action

    def test(self):
        episode_num = 5
        for i in range(episode_num):
            reward_sum = 0
            over = False
            state, _ = self.env.reset()
            while not over:
                action = self.get_action_by_max_p(state)
                next_state, reward, terminate, truncated, info = self.env.step(action)
                reward_sum += reward

                over = terminate or truncated
                state = next_state

            print("test {}: {}".format(i, reward_sum))

if __name__ == '__main__':
    # trainer = CartPoleTrainer(max_step=500, model_path="models/12.pth")
    # trainer.train()

    tester = CartPoleTester(model_path="models/12.pth")
    tester.test()