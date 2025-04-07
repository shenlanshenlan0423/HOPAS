# -*- coding: utf-8 -*-
"""
@Time ： 2025/1/17 21:55
@Auth ： Hongwei
@File ：CQL_DQN.py
@IDE ：PyCharm
"""
from definitions import *
from src.utils.rl_utils import VAnet, Qnet


class CQL_DQN(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, args):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.args = args
        self.tau = 0.005
        self.gamma = args.gamma
        self.alpha = 0.05  # Parameter for CQL regularization

        # CQL 无需使用dueling结构
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.Lr)  # args.Lr

    def take_action(self, state):
        q_values = self.q_net(state)
        action_prob = F.softmax(q_values, dim=1)  # stochastic policy
        action = action_prob.argmax(dim=1)
        max_action_q = q_values[range(len(q_values)), action]
        return q_values, max_action_q, action_prob

    def cql_loss(self, q_values_all_action, current_action):
        """Computes the CQL loss for a batch of Q-values and actions."""
        logsumexp = torch.logsumexp(q_values_all_action, dim=1, keepdim=True)
        q_a = q_values_all_action.gather(dim=1, index=current_action)
        return (logsumexp - q_a).mean()

    def soft_update(self, q_net, target_q_net):
        for target_param, param in zip(target_q_net.parameters(), q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def update(self, transition_dict):
        states = torch.cat(transition_dict['states'], dim=0)
        actions = torch.cat(transition_dict['actions'], dim=0)
        rewards = torch.cat(transition_dict['rewards'], dim=0)
        next_states = torch.cat(transition_dict['next_states'], dim=0)
        dones = torch.cat(transition_dict['dones'], dim=0)

        with torch.no_grad():
            max_next_q_values = self.target_q_net(next_states).max(dim=1, keepdim=True)[0]
            q_targets = rewards + self.gamma * max_next_q_values.squeeze(1) * (torch.ones_like(dones) - dones)

        q_values_all_action = self.q_net(states)
        argmax_action = actions.argmax(dim=1, keepdim=True)
        q_values = q_values_all_action.gather(dim=1, index=argmax_action)

        cql1_loss = self.cql_loss(q_values_all_action, current_action=argmax_action)
        bellman_error = F.mse_loss(q_values, q_targets)

        q1_loss = self.alpha * cql1_loss + bellman_error  # TODO  0.5 *

        self.optimizer.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.)  # TODO
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.q_net, self.target_q_net)
        return {"bellman_loss": bellman_error.detach().item(), "cql_loss": cql1_loss.detach().item(), "total_loss": q1_loss.detach().item()}


def training_CQL(agent, train_replay_buffer, args):
    losses = {'train': []}

    agent.train()
    for epoch in range(args.Epochs):
        # train_replay_buffer = copy.deepcopy(raw_replay_buffer)
        print('Batch number: {}\n'.format(train_replay_buffer.size() / args.Batch_size))
        with tqdm(total=int(train_replay_buffer.size() / args.Batch_size),
                  desc='Epoch {}/{}'.format(epoch + 1, args.Epochs)) as pbar:
            for i in range(int(train_replay_buffer.size() / args.Batch_size)):
                b_s, b_a, b_r, b_ns, b_d = train_replay_buffer.sample(args.Batch_size)  # 不放回采样
                transition_dict_for_train = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                loss_dict = agent.update(transition_dict_for_train)
                losses['train'].append(loss_dict['total_loss'])
                pbar.set_postfix({
                    'Train bellman_loss:': '{:.6f}'.format(loss_dict['bellman_loss']),
                    'Train cql_loss:': '{:.6f}'.format(loss_dict['cql_loss']),
                    'Train total_loss:': '{:.6f}'.format(loss_dict['total_loss']),
                })
                pbar.update(1)
    return agent, losses
