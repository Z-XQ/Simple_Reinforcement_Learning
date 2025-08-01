import gymnasium as gym
import torch
import torch.nn as nn
import os
import random

from tqdm import tqdm


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax()
        )

    def forward(self, x):
        return self.layer(x)


class CartPoleTrainer(object):
    def __init__(self, render_mode="rgb_array", max_step=500, model_path="models/1234.pth", lr=1e-3):
        self.env = gym.make("CartPole-v1", render_mode=render_mode, max_episode_steps=max_step)
        self.model = SimpleNet()
        self.lr = lr
        self.model_path = model_path
        save_dir = os.path.dirname(model_path)
        os.makedirs(save_dir, exist_ok=True)

    def get_action_by_p(self, state):
        """state: array. (4, )
        return. int. 0/1
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, 4)
        prob = self.model(state_tensor)
        action = random.choices(range(2), weights=prob[0].tolist(), k=1)[0]
        return action

    def get_episode(self):
        states, actions, rewards = [], [], []
        state, _ = self.env.reset()
        over = False
        while not over:
            # 策略模型，选择策略
            action = self.get_action_by_p(state)
            # 交互
            next_state, reward, terminate, truncated, info = self.env.step(action)
            # 记录
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            # next
            state = next_state
            over = terminate or truncated

        return states, actions, rewards

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        max_rewards = 0
        for epoch in tqdm(range(1000)):
            states, actions, rewards = self.get_episode()
            # 清空梯度
            optimizer.zero_grad()
            # 求梯度
            rewards_sum = 0
            for i in reversed(range(len(states))):
                # 状态si,ai对应的q值
                rewards_sum = rewards[i] + 0.99*rewards_sum
                # 策略概率 ln
                state_tensor = torch.tensor(states[i], dtype=torch.float32).reshape(1, 4)
                prob = self.model(state_tensor)
                cur_p = prob[0, actions[i]]
                # 最大化lnp * q，即最大化q值和选择此q值的概率
                loss = -cur_p.log() * rewards_sum
                loss.backward(retain_graph=True)
            # 利用梯度更新参数
            optimizer.step()
            # val
            if epoch % 100 == 0:
                val_result = sum([self.val_episode() for i in range(10)]) / 10
                print("average reward on val: {}".format(val_result))
                if val_result > max_rewards:
                    max_rewards = val_result
                    torch.save(self.model.state_dict(), self.model_path)

    def get_action_by_max(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, 4)
        prob = self.model(state_tensor)  # (1,2)
        action = torch.argmax(prob, dim=1).item()
        return action

    def val_episode(self):
        val_rewards = 0
        state, _ = self.env.reset()
        over = False
        while not over:
            # 根据策略模型，选出动作
            action = self.get_action_by_max(state)
            # 交互
            next_state, reward, terminate, truncated, info = self.env.step(action)
            val_rewards += reward
            # next
            over = terminate or truncated
            state = next_state

        return val_rewards


if __name__ == '__main__':
    trainer = CartPoleTrainer()
    trainer.train()
