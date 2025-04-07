# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/23 20:06
@Auth ： Hongwei
@File ：Clustering.py
@IDE ：PyCharm
"""
from definitions import *
from src.data.dataset_split import get_tensor_data
from src.model.D3QN import DuelingDoubleDQN
from sklearn.cluster import KMeans
from src.model.HOPAS import testing, compute_weight
from src.utils.rl_utils import prepare_replay_buffer, training_DQN


class HOPAS_wo_seq2seq:
    def __init__(self, args):
        self.args = args
        self.cluster_model = None
        self.agent_pool = {}

    def fit(self, df_train, data_path, args):
        gps = df_train.groupby('traj')
        # 每个患者使用其在ICU stay中的平均作为其state representation
        patient_represent_vec = np.array([gp.values[:, 2:-2].mean(axis=0).tolist() for _, gp in gps])
        self.cluster_model = KMeans(self.args.n_clusters, random_state=42)
        self.cluster_model.fit(patient_represent_vec)
        cluster_label = self.cluster_model.labels_
        pass
        # 0    3411
        # 2    3198
        # 1    2739

        patient_group_idx = list(gps.groups.keys())
        for subclass in range(args.n_clusters):
            # patients_index = np.where(cluster_label == subclass)[0]
            # sub_df = pd.concat([gps.get_group(patient_group_idx[id_]) for id_ in patients_index]).reset_index(drop=True)
            # states, acts, lengths, outcomes = get_tensor_data(sub_df['traj'].unique(), sub_df, outcome_col='rewards_90d')
            # torch.save((states, acts, lengths, outcomes), os.path.join(data_path, 'wo-seq2seq-class-{}-tuples'.format(subclass)))
            # sub_train_replay_buffer = prepare_replay_buffer(tensor_tuple=(states, acts, lengths, outcomes), args=args)
            # torch.save(sub_train_replay_buffer, data_path + 'wo-seq2seq-class-{}-train_replay_buffer'.format(subclass))
            sub_train_replay_buffer = torch.load(data_path + 'wo-seq2seq-class-{}-train_replay_buffer'.format(subclass))

            agent = DuelingDoubleDQN(state_dim=state_dim, action_dim=args.action_dim,
                                     hidden_dim=args.Hidden_size, gamma=args.gamma)
            agent, loss_dict = training_DQN(agent, sub_train_replay_buffer, args)
            self.agent_pool['class-{}'.format(subclass)] = agent
        return self

    def predict(self, df_test, transition_dict_for_test):
        gps = df_test.groupby('traj')
        lenss = [gp.shape[0] for _, gp in gps]
        patient_level_representations = np.array([gp.values[:, 2:-2].mean(axis=0).tolist() for _, gp in gps])
        extended_arr = []
        for i, repeat_count in enumerate(lenss):
            extended_arr.extend([patient_level_representations[i]] * (repeat_count - 1))  # length在RL的数据预处理时-1，所以此处也要-1
        extended_arr = np.array(extended_arr)
        point_to_center_distance = self.cluster_model.transform(extended_arr)
        return testing(self, transition_dict_for_test, point_to_center_distance)

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
