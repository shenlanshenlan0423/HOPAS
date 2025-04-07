# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/8 14:50
@Auth ： Hongwei
@File ：HOPAS.py
@IDE ：PyCharm
"""
from definitions import *
from src.utils.utils import prepare_tuples


class HOPAS:
    def __init__(self, data_path, encoder, cluster_model, agent_pool, args):
        self.data_path = data_path
        self.encoder = encoder
        self.cluster_model = cluster_model
        self.agent_pool = agent_pool
        self.args = args

    def get_stochastic_policy_with_agent_numb(self, test_data, transition_dict):
        test_set = prepare_tuples(test_data)
        test_loader = DataLoader(dataset=test_set, batch_size=self.args.batch_size, shuffle=False)
        # test set中所有患者的representation
        patient_level_representations = []
        lenss = []
        for (s, a, _, lens, _) in test_loader:
            time_mask = torch.zeros((s.shape[0], horizon), dtype=torch.bool)
            for i in range(s.shape[0]):
                time_mask[i, :lens[i]] = True
            _, attn_represent, _, _, _ = self.encoder.encoder(s, a, time_mask)
            patient_level_representations.extend(attn_represent.detach().cpu().numpy().tolist())
            lenss.extend(lens.squeeze(-1).detach().cpu().numpy().tolist())
        patient_level_representations = np.array(patient_level_representations)

        extended_arr = []
        for i, repeat_count in enumerate(lenss):
            extended_arr.extend([patient_level_representations[i]] * (repeat_count - 1))  # length在RL的数据预处理时-1，所以此处也要-1
        extended_arr = np.array(extended_arr)
        point_to_center_distance = self.cluster_model.transform(extended_arr)
        selected_action_with_agent_numb = np.array([testing(self, transition_dict, point_to_center_distance)[1].argmax(1).tolist(), point_to_center_distance.argmin(1).tolist()]).T
        return pd.DataFrame(selected_action_with_agent_numb, columns=['selected action', 'cluster label'])

    def test(self, test_data, transition_dict):
        test_set = prepare_tuples(test_data)
        test_loader = DataLoader(dataset=test_set, batch_size=self.args.batch_size, shuffle=False)
        # test set中所有患者的representation
        patient_level_representations = []
        lenss = []
        for (s, a, _, lens, _) in test_loader:
            time_mask = torch.zeros((s.shape[0], horizon), dtype=torch.bool)
            for i in range(s.shape[0]):
                time_mask[i, :lens[i]] = True
            _, attn_represent, _, _, _ = self.encoder.encoder(s, a, time_mask)
            patient_level_representations.extend(attn_represent.detach().cpu().numpy().tolist())
            lenss.extend(lens.squeeze(-1).detach().cpu().numpy().tolist())
        patient_level_representations = np.array(patient_level_representations)

        extended_arr = []
        for i, repeat_count in enumerate(lenss):
            extended_arr.extend([patient_level_representations[i]] * (repeat_count - 1))  # length在RL的数据预处理时-1，所以此处也要-1
        extended_arr = np.array(extended_arr)
        point_to_center_distance = self.cluster_model.transform(extended_arr)
        return testing(self, transition_dict, point_to_center_distance)


class HOPAS_A(HOPAS):
    def __init__(self, data_path, encoder, cluster_model, agent_pool, args):
        super().__init__(data_path, encoder, cluster_model, agent_pool, args)
        self.data_path = data_path
        self.encoder = encoder
        self.cluster_model = cluster_model
        self.agent_pool = agent_pool
        self.args = args

    # Strategy A: Single Well-matched Expert Decision-making.
    def take_action(self, states, agents_pool, distance):
        decision_prob = compute_weight(distance, self.args.lambda_)
        decision_agent = decision_prob.argmax(axis=1)
        agent1_index, agent2_index, agent3_index = np.where(decision_agent == 0)[0], np.where(decision_agent == 1)[0], np.where(decision_agent == 2)[0]

        Q_estimate_by_agent_1, _, agent_policy_by_agent_1 = agents_pool['class-0'].take_action(states)
        Q_estimate_by_agent_2, _, agent_policy_by_agent_2 = agents_pool['class-1'].take_action(states)
        Q_estimate_by_agent_3, _, agent_policy_by_agent_3 = agents_pool['class-2'].take_action(states)

        Q_estimate_list = [Q_estimate_by_agent_1, Q_estimate_by_agent_2, Q_estimate_by_agent_3]
        policy_list = [agent_policy_by_agent_1, agent_policy_by_agent_2, agent_policy_by_agent_3]
        idx_list = [agent1_index, agent2_index, agent3_index]
        Q_concat_list, policy_concat_list = [], []
        for idx in range(self.args.n_clusters):
            if Q_estimate_list[idx] == None:
                continue
            Q_concat_list.extend(np.hstack(
                (Q_estimate_list[idx][idx_list[idx]].detach().cpu().numpy().reshape(-1, action_dim), np.array(idx_list[idx]).reshape(-1, 1))))
            policy_concat_list.extend(np.hstack(
                (policy_list[idx][idx_list[idx]].detach().cpu().numpy().reshape(-1, action_dim), np.array(idx_list[idx]).reshape(-1, 1))))

        Q_estimate = pd.DataFrame(np.array(Q_concat_list)).set_index(action_dim).sort_index()
        agent_policy = pd.DataFrame(np.array(policy_concat_list)).set_index(action_dim).sort_index()
        return Q_estimate.values, agent_policy.values


class HOPAS_B(HOPAS):
    def __init__(self, data_path, encoder, cluster_model, agent_pool, args):
        super().__init__(data_path, encoder, cluster_model, agent_pool, args)
        self.data_path = data_path
        self.encoder = encoder
        self.cluster_model = cluster_model
        self.agent_pool = agent_pool
        self.args = args

    # Strategy B: Multi-experts Probabilistic Consensus Aggregation.
    def take_action(self, states, agents_pool, distance):
        decision_prob = compute_weight(distance, self.args.lambda_)

        Q_estimate_by_agent_1, _, agent_policy_by_agent_1 = agents_pool['class-0'].take_action(states)
        Q_estimate_by_agent_2, _, agent_policy_by_agent_2 = agents_pool['class-1'].take_action(states)
        Q_estimate_by_agent_3, _, agent_policy_by_agent_3 = agents_pool['class-2'].take_action(states)

        Q_estimate = ((Q_estimate_by_agent_1.detach().cpu().numpy() * decision_prob[:, 0, None]) +
                        (Q_estimate_by_agent_2.detach().cpu().numpy() * decision_prob[:, 1, None]) +
                        (Q_estimate_by_agent_3.detach().cpu().numpy() * decision_prob[:, 2, None]))

        agent_policy = ((agent_policy_by_agent_1.detach().cpu().numpy() * decision_prob[:, 0, None]) +
                        (agent_policy_by_agent_2.detach().cpu().numpy() * decision_prob[:, 1, None]) +
                        (agent_policy_by_agent_3.detach().cpu().numpy() * decision_prob[:, 2, None]))
        return Q_estimate, agent_policy


def testing(agent, transition_dict, point_to_center_distance):
    states = torch.cat(transition_dict['states'], dim=0)
    Q_estimate, agent_policy = agent.take_action(states, agent.agent_pool, point_to_center_distance)
    return Q_estimate, agent_policy, transition_dict


def compute_weight(distances, lambda_):
    distances = np.exp(distances * lambda_)
    weight = np.reciprocal(distances) / np.sum(np.reciprocal(distances), axis=1, keepdims=True)
    return weight
