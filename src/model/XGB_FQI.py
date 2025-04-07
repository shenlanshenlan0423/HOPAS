# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/16 15:23
@Auth ： Hongwei
@File ：XGB_FQI.py
@IDE ：PyCharm
"""
from definitions import *
from src.utils.utils import softmax


class XGBQLearner:
    def __init__(self):
        self.q_estimator = None

    def fit(self, transition_dict_for_train, args):
        states = torch.cat(transition_dict_for_train['states'], dim=0).cpu().numpy()
        actions = torch.cat(transition_dict_for_train['actions'], dim=0).cpu().numpy()
        rewards = torch.cat(transition_dict_for_train['rewards'], dim=0).cpu().numpy()
        next_states = torch.cat(transition_dict_for_train['next_states'], dim=0).cpu().numpy()

        max_q_values = np.zeros((states.shape[0]))  # Initialize Q
        i = np.hstack((states, actions))  # True state-action pair in training set
        action_space = pd.get_dummies(pd.Series(list(range(action_dim)))).values  # 所有可选动作 one-hot encoding
        i_t1 = np.hstack((np.repeat(next_states, action_dim, axis=0), np.tile(action_space, next_states.shape[0]).T))

        model = XGBRegressor(random_state=42)
        reward_mask = np.abs(rewards) < 12  # Note:terminal state的max_q_values没有意义（参考自DQN的Q更新公式）；否则reward上限的持续增加会影响训练的数值稳定性
        loss_res = []
        with tqdm(total=args.max_iteration) as pbar:
            for _ in range(args.max_iteration):
                o = rewards + args.gamma * (max_q_values * reward_mask)
                # sample_idx = np.random.choice(range(i.shape[0]), size=50000, replace=False)
                # model.fit(X=i[sample_idx], y=o[sample_idx])
                model.fit(X=i, y=o)
                q_values_hat = model.predict(i_t1).reshape(next_states.shape[0], action_dim)  # 估计next states在所有可选动作下的Q values
                mse_loss = np.mean((max_q_values - q_values_hat.max(axis=1)) ** 2)
                max_q_values = q_values_hat.max(axis=1)
                pbar.set_postfix({
                    'Loss': '{}'.format(mse_loss)
                })
                pbar.update(1)
                loss_res.append(mse_loss)
        self.q_estimator = model
        return self, loss_res

    def predict(self, transition_dict_for_test):
        states = torch.cat(transition_dict_for_test['states'], dim=0).cpu().numpy()
        action_space = pd.get_dummies(pd.Series(list(range(action_dim)))).values
        i_t1 = np.hstack((np.repeat(states, action_dim, axis=0), np.tile(action_space, states.shape[0]).T))

        Q_estimate = self.q_estimator.predict(i_t1).reshape(states.shape[0], action_dim)
        agent_policy = np.apply_along_axis(softmax, 1, Q_estimate)
        return Q_estimate, agent_policy

    def predict_Q(self, state_action):
        return self.q_estimator.predict(state_action)
