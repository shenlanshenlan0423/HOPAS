# -*- coding: utf-8 -*-
"""
@Auth ： Hongwei
@File ：dosage_diff_with_mortality.py
@IDE ：PyCharm
"""
from definitions import *
from src.main import parse_args
from src.model.HOPAS import HOPAS_A, HOPAS_B


def compute_mortality(diff_data, bin_number, boundary, bandwidth):
    mortality = []
    bins = np.linspace(-boundary, boundary, num=bin_number)
    for i, bin in enumerate(bins):
        if i == 0:
            outcome_array = diff_data[(diff_data[:, 0] < bin)][:, 1]
        elif i == len(bins) - 1:
            outcome_array = diff_data[(diff_data[:, 0] > bin)][:, 1]
        else:
            outcome_array = diff_data[(diff_data[:, 0] > bin - bandwidth) & (diff_data[:, 0] < bin + bandwidth)][:, 1]

        if len(outcome_array) != 0:
            mortality.append(len(outcome_array[outcome_array == -1]) / len(outcome_array))
        else:
            mortality.append(0)
    return list(bins), mortality


if __name__ == '__main__':
    # [-1, 0, 3600.0, 5402.161010742188, 7464.140504964193, 42645.373787434895]
    args = parse_args()
    action_dict = {0: 0, 1: 1800, 2: 4501.08, 3: 6433.15, 4: 25054.755}
    trained_agents = load_pickle(MODELS_DIR + '/trained_agents.pkl')
    trained_cluster_models = load_pickle(MODELS_DIR + '/Cluster_models.pkl')
    folds_plot_dict = {}
    for fold_idx in tqdm(range(folds)):
        fold_label = 'fold-{}'.format(fold_idx + 1)
        data_path = DATA_DIR + '/{}/{}/'.format('rewards_90d', fold_label)
        test_data = torch.load(data_path + '/test_set_tuples')
        test_replay_buffer = torch.load(data_path + 'test_set_tuples_replay_buffer')
        b_s, b_a, b_r, b_ns, b_d = test_replay_buffer.get_all_samples()
        transition_dict = {
            'states': b_s,
            'actions': b_a,
            'next_states': b_ns,
            'rewards': b_r,
            'dones': b_d
        }
        states = torch.cat(transition_dict['states'], dim=0)
        raw_action = torch.cat(transition_dict['actions'], dim=0).cpu().numpy().argmax(1)
        rewards = torch.cat(transition_dict['rewards'], dim=0).cpu().numpy()
        dones = torch.cat(transition_dict['dones'], dim=0).cpu().numpy()
        sample_size = states.shape[0]

        plot_dict = {}
        for i, model_name in enumerate(['HOPAS-A', 'HOPAS-B']):
            if model_name == 'HOPAS-A':
                model = HOPAS_A(data_path=data_path,
                                encoder=torch.load(MODELS_DIR + '/{}-{}.pt'.format(args.model_name, fold_label)),
                                cluster_model=trained_cluster_models[fold_label],
                                agent_pool=trained_agents[fold_label], args=args)
            elif model_name == 'HOPAS-B':
                model = HOPAS_B(data_path=data_path,
                                encoder=torch.load(MODELS_DIR + '/{}-{}.pt'.format(args.model_name, fold_label)),
                                cluster_model=trained_cluster_models[fold_label],
                                agent_pool=trained_agents[fold_label], args=args)
            Q_estimate, agent_policy, _ = model.test(test_data, transition_dict)
            agent_actions = agent_policy.argmax(1)

            # Note: compute the dosage diff
            Table_for_dosage_and_outcome = []
            for state_idx in range(sample_size):
                agent_action = action_dict[agent_actions[state_idx]]
                clin_action = action_dict[raw_action[state_idx]]
                Table_for_dosage_and_outcome.append([agent_action, clin_action])
            Table_for_dosage_and_outcome = pd.DataFrame(
                np.hstack((
                    np.array(Table_for_dosage_and_outcome),
                    rewards.reshape(-1, 1)
                )),
                columns=['Agent Heparin', 'Clin Heparin', 'Reward'])
            Table_for_dosage_and_outcome.iloc[dones != 1, -1] = np.nan
            Table_for_dosage_and_outcome.iloc[dones == 1, -1] = Table_for_dosage_and_outcome.iloc[
                                                                    dones == 1, -1] / np.abs(
                Table_for_dosage_and_outcome.iloc[dones == 1, -1].values)
            Table_for_dosage_and_outcome.iloc[:, -1].fillna(method='bfill', inplace=True)
            Table_for_dosage_and_outcome['dosage_diff'] = Table_for_dosage_and_outcome['Agent Heparin'].values - \
                                                          Table_for_dosage_and_outcome['Clin Heparin'].values

            sample_indices = np.where(dones == 1)[0]
            sample_index_list = []
            for i in range(len(sample_indices)):
                if i == 0:
                    start = 0
                else:
                    start = sample_indices[i - 1]
                end = sample_indices[i]
                for j in range(start, end):
                    sample_index_list.append(i)
            Table_for_dosage_and_outcome['traj'] = [0] + sample_index_list

            gps = Table_for_dosage_and_outcome.groupby('traj')
            Table_for_dosage_diff = []
            for patient_idx, group in gps:
                Table_for_dosage_diff.append([group['dosage_diff'].mean(), group['Reward'].mean()])
            Table_for_dosage_diff = pd.DataFrame(np.array(Table_for_dosage_diff), columns=['dosage_diff', 'Reward'])

            bin_number_, bandwidth_ = 20, 1000
            bins, diff_mortality = compute_mortality(Table_for_dosage_diff[['dosage_diff', 'Reward']].values,
                                                     bin_number=bin_number_, boundary=8000, bandwidth=bandwidth_)
            plot_dict[model_name] = {'x_bins': bins, 'mortality': diff_mortality}
        folds_plot_dict[fold_idx] = plot_dict

    save_pickle(folds_plot_dict, RESULT_DIR + 'new_dosage_diff_with_mortality.pkl')
