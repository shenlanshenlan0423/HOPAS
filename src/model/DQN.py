# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/28 10:37
@Auth ： Hongwei
@File ：DQN.py
@IDE ：PyCharm
"""
from definitions import *
from src.utils.rl_utils import Qnet


class DQN(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, gamma):
        super(DQN, self).__init__()
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.gamma = gamma
        self.target_update = 3
        self.count = 0

    def take_action(self, state):
        q_values = self.q_net(state)
        action_prob = F.softmax(q_values, dim=1)  # stochastic policy
        action = action_prob.argmax(dim=1)
        max_action_q = q_values[range(len(q_values)), action]
        return q_values, max_action_q, action_prob

    def update(self, transition_dict):
        states = torch.cat(transition_dict['states'], dim=0)
        actions = torch.cat(transition_dict['actions'], dim=0)
        rewards = torch.cat(transition_dict['rewards'], dim=0)
        next_states = torch.cat(transition_dict['next_states'], dim=0)
        dones = torch.cat(transition_dict['dones'], dim=0)

        argmax_action = actions.argmax(dim=1, keepdim=True)
        q_values = self.q_net(states).gather(dim=1, index=argmax_action)
        max_next_q_values = self.target_q_net(next_states).max(dim=1, keepdim=True)[0]

        q_targets = rewards + self.gamma * max_next_q_values.squeeze(1) * (torch.ones_like(dones) - dones)
        if self.count % self.target_update == 0:  # 每self.target_update更新一次targetQ的参数
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
        return q_values, q_targets.unsqueeze(1)
