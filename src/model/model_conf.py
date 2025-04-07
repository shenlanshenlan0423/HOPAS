# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/27 18:35
@Auth ： Hongwei
@File ：model_conf.py
@IDE ：PyCharm
"""
from definitions import *
from src.model.CQL_DQN import CQL_DQN, training_CQL
from src.model.D3QN import DuelingDoubleDQN
from src.model.DQN import DQN
from src.model.DoubleDQN import DoubleDQN
from src.model.DuelingDQN import DuelingDQN
from src.model.HOPAS_wo_seq2seq import HOPAS_wo_seq2seq
from src.model.WD3QNE import WD3QNE, train_WD3QNE
from src.model.XGB_FQI import XGBQLearner
from src.utils.rl_utils import training_DQN


class Configs:
    def __init__(self, args):
        self.args = args
        self.DQN = DQN(state_dim=state_dim,
                       action_dim=action_dim,
                       hidden_dim=args.Hidden_size,
                       gamma=args.gamma)
        self.DoubleDQN = DoubleDQN(state_dim=state_dim,
                                   action_dim=action_dim,
                                   hidden_dim=args.Hidden_size,
                                   gamma=args.gamma)
        self.DuelingDQN = DuelingDQN(state_dim=state_dim,
                                     action_dim=action_dim,
                                     hidden_dim=args.Hidden_size,
                                     gamma=args.gamma)
        self.DuelingDoubleDQN = DuelingDoubleDQN(state_dim=state_dim,
                                                 action_dim=action_dim,
                                                 hidden_dim=args.Hidden_size,
                                                 gamma=args.gamma)
        self.DQN_CQL = CQL_DQN(state_dim=state_dim,
                               hidden_dim=args.Hidden_size,
                               action_dim=action_dim, args=args)
        # IEEE Transactions on Neural Networks and Learning Systems 2024 10.1109/TNNLS.2022.3176204
        self.XGB_FQI = XGBQLearner()
        # NPJ 10.1038/s41746-023-00755-5
        self.WD3QNE = WD3QNE(state_dim=state_dim,
                             action_dim=action_dim,
                             hidden_dim=args.Hidden_size,
                             gamma=args.gamma)
        self.HOPAS_wo_seq2seq = HOPAS_wo_seq2seq(args)

    def train_DQN(self, agent, train_replay_buffer):
        return training_DQN(agent, train_replay_buffer, self.args)

    def test_DQN(self, agent, transition_dict):
        agent.eval()
        states = torch.cat(transition_dict['states'], dim=0)
        Q_estimate, _, agent_policy = agent.take_action(states)
        agent_policy = agent_policy.detach().cpu().numpy()
        return Q_estimate.detach().cpu().numpy(), agent_policy, transition_dict  # stochastic policy

    def train_QNE(self, agent, train_replay_buffer):
        return train_WD3QNE(agent, train_replay_buffer, self.args)

    def test_QNE(self, agent, transition_dict):
        agent.eval()
        states = torch.cat(transition_dict['states'], dim=0)
        Q_estimate, _, agent_policy = agent.take_action(states)
        agent_policy = agent_policy.detach().cpu().numpy()
        return Q_estimate.detach().cpu().numpy(), agent_policy, transition_dict

    def train_CQL(self, agent, train_replay_buffer):
        return training_CQL(agent, train_replay_buffer, self.args)

    def test_CQL(self, agent, transition_dict):
        agent.eval()
        states = torch.cat(transition_dict['states'], dim=0)
        Q_estimate, _, agent_policy = agent.take_action(states)
        agent_policy = agent_policy.detach().cpu().numpy()
        return Q_estimate.detach().cpu().numpy(), agent_policy, transition_dict
