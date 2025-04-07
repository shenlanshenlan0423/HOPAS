# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/24 15:46
@Auth ： Hongwei
@File ：WD3QNE.py
@IDE ：PyCharm
"""
from definitions import *
from src.utils.rl_utils import VAnet


class WD3QNE(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, gamma):
        super(WD3QNE, self).__init__()
        self.action_dim = action_dim
        self.q_net = VAnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net = VAnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.gamma = gamma
        self.target_update = 3
        self.count = 0

    def take_action(self, state):
        q_values = self.q_net(state)
        action_prob = F.softmax(q_values, dim=1)
        action = action_prob.argmax(dim=1)
        max_action_q = q_values[range(len(q_values)), action]
        return q_values, max_action_q, action_prob

    def update(self, transition_dict):
        states = torch.cat(transition_dict['states'], dim=0)
        actions = torch.cat(transition_dict['actions'], dim=0)
        rewards = torch.cat(transition_dict['rewards'], dim=0)
        next_states = torch.cat(transition_dict['next_states'], dim=0)
        dones = torch.cat(transition_dict['dones'], dim=0)

        # SOFA 5: -0.56522; 7: -0.391304; 最后一列为SOFA
        # 脓毒症低风险人群（SOFA评分≤7分）抗凝治疗无明显获益  许伟伟,李明,崔广清,等. 脓毒症抗凝治疗的意义与未来[J]. 中华危重病急救医学,2021,33(05)：621-625.
        no_heparin_index = torch.where(states[:, -1] < -0.391304)[0]

        argmax_action = actions.argmax(dim=1, keepdim=True)
        q_values = self.q_net(states).gather(dim=1, index=argmax_action)

        # Note: DoubleDQN: 利用两套独立训练的神经网络，以缓解Q值过高估计的问
        max_action = self.q_net(next_states).argmax(dim=1, keepdim=True)
        target_q = self.target_q_net(next_states)
        max_next_q_values_1 = target_q.gather(dim=1, index=max_action)
        phi = max_next_q_values_1.squeeze(1) / target_q.sum(1)
        # Note: DuelingDQN
        max_next_q_values_2 = target_q.max(dim=1, keepdim=True)[0]
        sigma = max_next_q_values_2.squeeze(1) / target_q.sum(1)
        p = phi / (phi + sigma)
        max_next_q_values = p.unsqueeze(1) * max_next_q_values_1 + (1 - p).unsqueeze(1) * max_next_q_values_2
        # BUG: human-in-loop 在SIT这个context中, SOFA<5的患者采用医生的heparin决策这个结论不一定成立
        # max_next_q_values[human_expert_decide_rows] = q_values[human_expert_decide_rows]
        max_next_q_values[no_heparin_index] = self.q_net(states).gather(dim=1, index=torch.zeros_like(argmax_action))[no_heparin_index]

        q_targets = rewards + self.gamma * max_next_q_values.squeeze(1) * (torch.ones_like(dones) - dones)
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
        return q_values, q_targets.unsqueeze(1)


def train_WD3QNE(agent, raw_replay_buffer, args):
    # The original author proposed the use of Huber loss.
    # In our implementation, for the purpose of facilitating comparison, we have uniformly adopted the MSE loss.
    # criterion = nn.HuberLoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(agent.parameters(), lr=args.Lr)

    losses = {'train': []}
    agent.train()
    for epoch in range(args.Epochs):
        train_replay_buffer = copy.deepcopy(raw_replay_buffer)
        with tqdm(total=int(train_replay_buffer.size() / args.Batch_size), desc='Epoch {}/{}'.format(epoch + 1, args.Epochs)) as pbar:
            for i in range(int(train_replay_buffer.size() / args.Batch_size)):
                optimizer.zero_grad()
                b_s, b_a, b_r, b_ns, b_d = train_replay_buffer.sample(args.Batch_size)  # 不放回采样
                transition_dict_for_train = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                q_values, q_targets = agent.update(transition_dict_for_train)
                loss = criterion(q_values, q_targets.detach())
                loss.backward()
                optimizer.step()

                losses['train'].append(loss.item())
                pbar.set_postfix({
                    'Train MSE:': '{:.6f}'.format(loss.item()),
                })
                pbar.update(1)
    return agent, losses
