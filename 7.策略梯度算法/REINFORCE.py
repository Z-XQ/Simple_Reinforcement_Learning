import os

import gymnasium as gym
import torch
import random

from torch import nn
from tqdm import tqdm

# ======================== 1. 定义环境（GridWorldEnv 改为 CartPole 包装器） ========================
class MyEnvWrapper(gym.Wrapper):
    def __init__(self, render_mode='rgb_array', max_steps=200):
        # 创建 CartPole-v1 环境，渲染模式设为 rgb_array（用于图像输出，若不需要可视化可简化）
        env = gym.make('CartPole-v1', render_mode=render_mode)  # rgb_array, human
        super().__init__(env)
        self.env = env
        self.cur_step_n = 0  # 记录当前回合步数，用于自定义终止条件
        self.max_steps = max_steps  # 最大迭代步长

    def reset(self, **kwargs):
        # 重置环境，返回初始状态，获取一个新的episode
        state, _ = self.env.reset()
        # 重置步数计数
        self.cur_step_n = 0
        return state

    def step(self, action):
        # 执行一步动作，返回环境交互结果
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated  # 自带的 综合终止标志
        self.cur_step_n += 1  # 步数+1

        # 自定义终止条件：若步数超过 max_steps，强制结束回合
        if self.cur_step_n >= self.max_steps:
            done = True

        return state, reward, done, info


# ======================== 2. 定义模型与动作获取逻辑 ========================
# 构建简单的策略网络（输入状态维度为 4，输出 2 个动作的概率）
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)  # 转为概率分布（dim=1 对动作维度做归一化）
        )

    def forward(self, x):
        return self.layers(x)


class CartPoleTrainer:
    """CartPole-v1环境的策略梯度训练器"""

    def __init__(self, render_mode='rgb_array', max_steps=200, learning_rate=1e-3, save_model_path=""):
        """
        初始化训练器

        参数:
            render_mode: 渲染模式，'rgb_array'或'human'
            max_steps: 每局最大步数
            learning_rate: 学习率
        """
        # 初始化环境
        self.env = MyEnvWrapper(render_mode, max_steps)
        self.learning_rate = learning_rate
        self.save_model_path = save_model_path
        # 初始化policy
        self.model = SimpleModel()
        # 确保模型保存目录存在
        save_dir = os.path.dirname(save_model_path)
        os.makedirs(save_dir, exist_ok=True)

    # 定义动作获取函数：根据状态和当前策略模型，输出动作概率，再采样动作
    def get_action(self, state):
        # 将状态转为张量，调整形状为 [1, 4]（适配模型输入）
        state = torch.FloatTensor(state).reshape(1, 4)

        # 模型输出动作概率分布 [1, 2]
        prob = self.model(state)  # (1,4) -> (1,2)

        # 因为是on-policy，所以需要多些探索，使得生成的数据更加多样。
        # 根据概率采样动作（动作概率列表[0, 1]，weights[0.7, 0.3]为概率列表，k=1 选 1 个动作）
        action = random.choices(range(2), weights=prob[0].tolist(), k=1)[0]
        return action  # 返回当前策略下，一个新动作的索引

    # ======================== 3. 定义数据采集函数（获取一局游戏数据） ========================
    def get_episode(self):
        states = []   # 存储每一步的状态
        rewards = []  # 存储每一步的奖励
        actions = []  # 存储每一步的动作
        state = self.env.reset()  # 重置环境，随机一个初始状态，开始新回合
        over = False         # 标记回合是否结束
        while not over:
            action = self.get_action(state)  # 根据当前状态选动作
            # 执行动作，获取下一状态、奖励、结束标志
            next_state, reward, over, _ = self.env.step(action)
            # 记录交互数据
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            state = next_state  # 更新状态，进入下一循环
        return states, rewards, actions

    # ======================== 4. 定义训练函数（核心逻辑：策略梯度更新） ========================
    def train(self):
        # 使用 Adam 优化器，学习率设为 1e-3
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        max_return = 0
        # 训练 1000 轮（每轮玩一局游戏并更新模型），即1000个episode
        for epoch in tqdm(range(1000)):
            # 获取一局游戏的数据（状态、奖励、动作）episode
            states, rewards, actions = self.get_episode()

            optimizer.zero_grad()  # 清空梯度
            reward_sum = 0         # 初始化累计奖励（用于反向传播）
            loss_sum = 0
            # 逆序遍历状态（从最后一步往前，符合策略梯度的时序更新逻辑），i是从大到小
            for i in reversed(range(len(states))):
                # 更新累计奖励（带贴现，这里简化为固定系数 0.98，实际应该是0.98^epoch），q-value
                reward_sum = reward_sum * 0.99 + rewards[i]

                # 计算状态对应的动作概率
                state = torch.FloatTensor(states[i]).reshape(1, 4)
                action = actions[i]
                prob = self.model(state)  # 输出 [1, 2]
                prob = prob[0, action]  # 提取当前动作对应的概率（标量）

                # 构造损失函数：负对数概率 * 累计奖励（最小化 loss 就等价于最大化原目标prob.log() * reward_sum）
                loss = -prob.log() * reward_sum
                # 需要对同一个计算图进行多次反向传播（累积每一步的梯度），通过保留计算图，确保后续的反向传播能正常执行，避免因计算图被销毁而报错。
                # 我们需要将每一步的梯度累积起来（而非覆盖），最后通过 optimizer.step() 一次性更新参数。
                loss.backward(retain_graph=True)  # 反向传播，保留计算图（因多次反向传播）
                loss_sum += loss.item()

            optimizer.step()  # 更新模型参数

            # print("loss: ", loss_sum / len(states))

            # 每 100 轮测试一次（跑 10 局取平均，评估性能）
            if epoch % 100 == 0:
                # 测试 10 局，取平均奖励
                val_result = sum([self.val() for _ in range(10)]) / 10
                # 打印轮次与测试结果
                print(f"Epoch: {epoch}, Test Reward: {val_result}")

            """保存模型参数"""
            if reward_sum > max_return:
                max_return = reward_sum
                torch.save(self.model.state_dict(), self.save_model_path)
                print(f"模型已保存至: {self.save_model_path}")

    # ======================== 5. 定义测试函数（评估当前策略性能） ========================
    def val(self):
        state = self.env.reset()  # 重置环境，随机一个初始状态
        reward_sum = 0       # 累计奖励
        over = False         # 标记是否结束，停止条件：自带条件和达到自定义的最大迭代次数
        while not over:
            action = self.get_action(state)  # 基于当前的神经网络，选动作
            # 执行动作，更新状态与奖励
            state, reward, over, _ = self.env.step(action)
            reward_sum += reward  # 累加奖励
        return reward_sum  # 返回单局总奖励

# ======================== 4. 测试类 ========================
class CartPoleTester:
    """使用保存的模型测试REINFORCE算法"""

    def __init__(self, model_path="models/cartpole_reinforce.pth", render_mode='human'):
        self.model_path = model_path
        self.env = MyEnvWrapper(render_mode=render_mode, max_steps=500)  # 测试时允许更长步数
        self.model = SimpleModel()  # 创建相同结构的模型
        self._load_model()  # 加载训练好的参数

    def _load_model(self):
        """加载模型参数"""
        try:
            self.model.load_state_dict(torch.load(self.model_path))
            print(f"成功加载模型: {self.model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise

    def get_action(self, state):
        """使用模型获取动作"""
        state = torch.FloatTensor(state).reshape(1, 4)
        with torch.no_grad():  # 测试时不需要计算梯度
            prob = self.model(state)
        action = torch.argmax(prob).item()  # 测试时使用确定性策略（选择概率最高的动作）
        return action

    def test(self, episodes=5):
        """运行测试"""
        print(f"\n开始测试，共运行 {episodes} 局...")

        total_reward = 0
        for episode in range(episodes):
            state = self.env.reset()
            reward_sum = 0
            over = False
            step = 0

            while not over:
                action = self.get_action(state)
                state, reward, over, _ = self.env.step(action)
                reward_sum += reward
                step += 1

                # 显示每步信息
                if step % 50 == 0:
                    print(f"Episode {episode + 1}, Step {step}, Current Reward: {reward_sum}")

            total_reward += reward_sum
            print(f"第 {episode + 1} 局结束，奖励: {reward_sum}")

        avg_reward = total_reward / episodes
        print(f"\n测试完成，平均奖励: {avg_reward:.2f}")
        return avg_reward


if __name__ == "__main__":
    # train
    trainer = CartPoleTrainer(render_mode="rgb_array", max_steps=500, learning_rate=1e-3,
                              save_model_path="models/cartpole_reinforce.pth")
    trainer.train()

    # 使用训练好的模型进行测试
    tester = CartPoleTester(render_mode='human')  # 'human'模式用于可视化
    tester.test(episodes=5)
