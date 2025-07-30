import os
import random

import gymnasium as gym
import torch
from torch import nn
from tqdm import tqdm


# ======================== 1. 定义环境 ========================
class MyEnvWrapper(gym.Wrapper):
    def __init__(self, render_mode='rgb_array', max_steps=200):
        # 创建 CartPole-v1 环境，渲染模式设为 rgb_array（用于图像输出，若不需要可视化可简化）
        env = gym.make('CartPole-v1', max_steps=max_steps, render_mode=render_mode)  # rgb_array, human
        super().__init__(env)
        self.env = env

    def reset(self, **kwargs):
        # 重置环境，返回初始状态，获取一个新的episode
        state, _ = self.env.reset()
        return state

    def step(self, action):
        # 执行一步动作，返回环境交互结果
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated  # 自带的 综合终止标志

        return state, reward, done, info


# ======================== 2. 定义策略模型 ========================
class SimpleModel(nn.Module):
    def  __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)


class CartPoleTrainer:
    def __init__(self, render_mode="rgb_array", max_steps=200,
                 learning_rate=1e-3,
                 save_weight_path="1.pth"):
        self.env = gym.make("CartPole-v1", max_episode_steps=max_steps, render_mode=render_mode)
        self.learning_rate = learning_rate
        self.model = SimpleModel()

        save_dir = os.path.dirname(save_weight_path)
        os.makedirs(save_dir, exist_ok=True)
        self.save_weight_path = save_weight_path

    def get_action(self, state):
        """
        a = model(s)
        :param state: numpy. float32. (4,)
        :return: int. 0/1
        """
        state_tensor = torch.FloatTensor(state).reshape(1, 4)
        prob = self.model(state_tensor)
        action = random.choices(range(2), weights=prob[0].tolist(), k=1)[0]
        return action

    def get_action_by_max_p(self, state):
        state_tensor = torch.FloatTensor(state).reshape(1, 4)
        prob = self.model(state_tensor)
        # 最大概率所在的索引
        action = torch.argmax(prob, 1).item()
        return action

    def get_episode(self):
        states = []   # 存储每一步的状态
        actions = []  # 存储每一步的动作
        rewards = []  # 存储每一步的奖励

        state, info = self.env.reset()  # 重置环境，智能体回到初始状态并返回。
        over = False
        while not over:
            # 当前状态下array(4,)，获取策略int(0/1)
            action = self.get_action(state)

            # 输入动作到环境中，执行
            next_state, reward, terminated, truncated, info = self.env.step(action)

            # 存储
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # 更新状态，进入下一循环
            state = next_state
            over = terminated or truncated

        return states, actions, rewards

    def train(self):
        # adam优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        max_return = 0  # 大于此值，存放权重
        for epoch in tqdm(range(1000)):
            # 1, 获取episode数据
            states, actions, rewards = self.get_episode()

            optimizer.zero_grad()  # 清空梯度

            reward_sum = 0  # 初始化累计奖励
            # loss_sum = 0
            for i in reversed(range(len(states))):
                # 2, q(si,ai)=reward_sum，近似动作价值，用于反向传播
                reward_sum = reward_sum * 0.99 + rewards[i]

                # 3, 动作概率Π(a|s)
                # input data
                state = torch.FloatTensor(states[i]).reshape(1, 4)
                output_all_prob = self.model(state)  # 输出全部动作概率，(1,2)
                cur_prob = output_all_prob[0, actions[i]]

                # 4，求loss，最大化累计期望奖励（最优策略），选择动作log概率 * 选择此动作的回报，目的是最大化 “高价值动作被选中的概率”
                loss = -cur_prob.log() * reward_sum
                # loss_sum += loss.item()

                # 只会累积每一步的梯度，不会更新参数，最后通过 optimizer.step() 一次性更新参数。
                loss.backward(retain_graph=True)

            # 更新模型参数
            optimizer.step()
            # print("loss: ", loss_sum/len(states))

            # 每 100 轮测试一次（跑 10 局取平均，评估性能）
            if epoch % 100 == 0:
                # 测试 10 局，取平均奖励
                val_result = sum([self.val_episode() for _ in range(10)]) / 10
                # 打印轮次与测试结果
                print(f"Epoch: {epoch}, Test Reward: {val_result}")

            """保存模型参数"""
            if reward_sum > max_return:
                max_return = reward_sum
                torch.save(self.model.state_dict(), self.save_weight_path)
                # print(f"模型已保存至: {self.save_weight_path}")

    def val_episode(self):
        # 重置环境，随机一个初始状态
        state, info = self.env.reset()

        reward_sum = 0  # 计算总分，走一步没结束则加一分
        over = False
        while not over:
            # 输出策略
            action = self.get_action_by_max_p(state)

            # 交互
            state, reward, terminated, truncated, info = self.env.step(action)
            reward_sum += reward

            over = terminated or truncated

        return reward_sum


class CartPoleTester:
    def __init__(self, model_path="1.pth", render_mode="human"):
        self.model_path = model_path
        self.model = SimpleModel()
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

if __name__ == "__main__":
    # train
    trainer = CartPoleTrainer(render_mode="rgb_array", max_steps=500, learning_rate=1e-3,
                              save_weight_path="models/cartpole_reinforce.pth")
    trainer.train()

    # 使用训练好的模型进行测试
    tester = CartPoleTester(model_path="models/cartpole_reinforce.pth", render_mode='human')  # 'human'模式用于可视化
    tester.test(episode_num=5)