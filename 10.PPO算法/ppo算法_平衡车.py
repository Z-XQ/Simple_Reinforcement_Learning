import random
import gymnasium as gym
import torch
import torch.nn as nn
from tqdm import tqdm


#定义模型
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, state):
        """
        Args:
            state: tensor. (4, )
        Returns:
            actor: p = actor_net(s)
        """
        p = self.layer(state)  # (4, ) -> (2, )
        return p


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        """
        Args:
            state: tensor. (4, )
        Returns:
            actor: val = net(s)
        """
        p = self.layer(state)  # (4, ) -> (1, )
        return p


class CartPoleTrainer(object):
    def __init__(self):
        self.env = gym.make("CartPole-v1")

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

    # 优势函数
    @staticmethod
    def get_advantages(deltas):
        """
        inputs: episode内部每个状态的td error
        outputs: A是td error的加权和
        """
        advantages = []

        # 反向遍历deltas
        s = 0.0  # 累计和
        for delta in deltas[::-1]:  # 逆向累加
            s = 0.98 * 0.95 * s + delta  # 累计td error
            advantages.append(s)  # 当前时刻的a值放在后面

        # 逆序
        advantages.reverse()
        return advantages

    # 得到一个动作
    def get_action(self, state):  # array, (4, )
        state = torch.tensor(state, dtype=torch.float32).reshape(1, 4)
        prob = self.actor(state)  # [1, 4] -> [1, 2]

        # 根据概率选择一个动作
        action = random.choices(range(2), weights=prob[0].tolist(), k=1)[0]
        return action

    # 得到一个动作
    def get_action_by_max_p(self, state):  # array, (4, )
        state = torch.tensor(state, dtype=torch.float32).reshape(1, 4)
        prob = self.actor(state)  # [1, 4] -> [1, 2]

        # 根据概率选择一个动作
        action = torch.argmax(prob, dim=1).item()
        return action

    def get_data(self):
        """

        Returns:
            states: [b, 4]
            rewards: [b, 1]
            actions: [b, 1]
            next_states: [b, 4]
            overs: [b, 1]
        """
        next_states = []
        states = []
        actions = []
        rewards = []
        overs = []

        # 初始化游戏
        state, _ = self.env.reset()

        # 玩到游戏结束为止
        over = False
        while not over:
            # 根据当前状态得到一个动作
            action = self.get_action(state)

            # 执行动作,得到反馈
            next_state, reward, terminate, truncated, _ = self.env.step(action)
            over = terminate or truncated

            # 记录数据样本
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            overs.append(over)

            # 更新游戏状态,开始下一个动作
            state = next_state

        states = torch.FloatTensor(states).reshape(-1, 4)  # [b, 4]
        rewards = torch.FloatTensor(rewards).reshape(-1, 1)  # [b, 1]
        actions = torch.LongTensor(actions).reshape(-1, 1)  # [b, 1]
        next_states = torch.FloatTensor(next_states).reshape(-1, 4)  # [b, 4]
        overs = torch.LongTensor(overs).reshape(-1, 1)  # [b, 1]
        return states, rewards, actions, next_states, overs

    def train(self):
        optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-2)
        loss_fn = torch.nn.MSELoss()

        # 玩N局游戏,每局游戏训练M次
        for epoch in tqdm(range(1000)):
            # 1. 玩一局游戏,得到数据
            states, rewards, actions, next_states, overs = self.get_data()

            # 2. TD Error
            # 计算values
            values = self.critic(states)  # [b, 4] -> [b, 1]
            # 计算td target: r + γ*net(s')
            targets = self.critic(next_states).detach()  # [b, 4] -> [b, 1]
            targets = targets * 0.98
            targets *= (1 - overs)  # t = t*(1-over) 将over估计分数置为0
            targets += rewards  # t = r + t
            # [b, 1]. TD Error
            deltas = (targets - values).squeeze(dim=1).tolist()

            # 3. 计算优势,这里的advantages有点像是策略梯度里的reward_sum, 只是这里计算的不是reward,而是target和value的差
            advantages = self.get_advantages(deltas)  # list of float.
            advantages = torch.tensor(advantages, dtype=torch.float32).reshape(-1, 1)  # (b,1)

            # 4. 取出每一步动作的概率
            # [b, 2] -> [b, 1]
            old_probs = self.actor(states)  # [b, 2]
            old_probs = old_probs.gather(dim=1, index=actions)  # actions.shape=[b, 1]
            old_probs = old_probs.detach()  # [b, 1]

            # 5. 每批数据反复训练10次
            for _ in range(10):
                # 5.1 重新计算每一步动作的概率
                new_probs = self.actor(states)  # [b, 4] -> [b, 2]
                new_probs = new_probs.gather(dim=1, index=actions)  # [b, 2] -> [b, 1]
                new_probs = new_probs  # [b, 1]

                # 5.2 策略损失
                # 求出概率的变化ratio
                ratios = new_probs / old_probs  # [b, 1] - [b, 1] -> [b, 1]

                # 计算截断的和不截断的两份loss,取其中小的
                surr1 = ratios * advantages  # [b, 1] * [b, 1] -> [b, 1]
                surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages  # [b, 1] * [b, 1] -> [b, 1]
                loss_actor = -torch.min(surr1, surr2)  # [b, 1] 每个样本的策略损失
                loss_actor = loss_actor.mean()  # batch平均策略损失

                # 5.3 价值损失
                # 必须重新计算value, 并计算时序差分loss
                values = self.critic(states)  # 在不同的net计算loss，这里是loss_critic
                loss_critic = loss_fn(values, targets)  # loss = net(s) - (r + γ*net(s'))

                # 5.4 更新参数
                optimizer_actor.zero_grad()
                loss_actor.backward()
                optimizer_actor.step()

                optimizer_critic.zero_grad()
                loss_critic.backward()
                optimizer_critic.step()

            if epoch % 50 == 0:
                test_result = sum([self.test() for _ in range(10)]) / 10
                print(epoch, test_result)

    def test(self):
        # 初始化游戏
        state, _ = self.env.reset()

        # 记录反馈值的和,这个值越大越好
        reward_sum = 0

        # 玩到游戏结束为止
        over = False
        while not over:
            # 根据当前状态得到一个动作
            # action = self.get_action_by_max_p(state)
            action = self.get_action(state)
            # 执行动作,得到反馈
            state, reward, terminate, truncated, _ = self.env.step(action)
            over = terminate or truncated
            reward_sum += reward

        return reward_sum


if __name__ == '__main__':
    trainer = CartPoleTrainer()
    trainer.train()