# -*- coding: utf-8 -*-
"""
@Auth ： Hongwei
@File ：dataset_split.py
@IDE ：PyCharm
"""
from definitions import *
from src.utils.rl_utils import prepare_replay_buffer


def get_tensor_data(traj_idxs, df, outcome_col):
    data_trajectory = {}
    data_trajectory['traj'] = {}

    for i in tqdm(traj_idxs):
        traj_i = df[df['traj'] == i].sort_values(by='step')
        data_trajectory['traj'][i] = {}
        data_trajectory['traj'][i]['states'] = torch.Tensor(traj_i[state_cols].values).to(device)
        data_trajectory['traj'][i]['actions'] = torch.Tensor(traj_i[action_col].values.astype(np.int32)).to(device).long()
        data_trajectory['traj'][i]['outcome'] = torch.Tensor(traj_i[outcome_col].values).to(device)

    states = torch.zeros((len(traj_idxs), horizon, len(state_cols)))
    actions = torch.zeros((len(traj_idxs), horizon, action_dim))
    action_temp = torch.eye(action_dim).to(device)
    lengths = torch.zeros((len(traj_idxs)), dtype=torch.int)
    outcomes = torch.zeros((len(traj_idxs), horizon))
    for ii, traj in enumerate(traj_idxs):
        state = data_trajectory['traj'][traj]['states']
        length = state.shape[0]
        lengths[ii] = length
        action = data_trajectory['traj'][traj]['actions'].view(-1, 1)
        temp = action_temp[action].squeeze(1)
        actions[ii] = torch.cat((temp, torch.zeros((horizon - length, action_dim), dtype=torch.float).to(device)))
        outcome = data_trajectory['traj'][traj]['outcome']
        states[ii] = torch.cat((state, torch.zeros((horizon - length, state.shape[1]), dtype=torch.float).to(device)))
        outcomes[ii] = torch.cat((outcome, torch.zeros((horizon - length), dtype=torch.float).to(device)))

    # Note: 去掉只有一个时刻的患者的数据
    states = states[lengths > 1.0].to(device)
    actions = actions[lengths > 1.0].to(device)
    outcomes = outcomes[lengths > 1.0].to(device)
    lengths = lengths[lengths > 1.0].to(device)
    return states, actions, lengths, outcomes


def split_sepsis_cohort(reward_label):
    if reward_label == 'rewards_90d':
        fullzs_data_path = os.path.join(DATA_DIR, 'mimic-iv-sepsis_withTimes.csv')
    elif reward_label == 'rewards_icu':
        fullzs_data_path = os.path.join(DATA_DIR, 'eICU-sepsis_withTimes.csv')

    SAVE_DIR = DATA_DIR + '/{}/'.format(reward_label)
    os.makedirs(SAVE_DIR, exist_ok=True)

    full_zs = pd.read_csv(fullzs_data_path)
    # patient-level
    temp = full_zs.groupby('traj')[reward_label].sum()
    X = temp.index.values
    y = temp.values

    all_cols = ['traj', 'step'] + state_cols + action_col + [reward_label]
    df_scaled = full_zs[all_cols].copy()
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(df_scaled[temporal_cols])
    df_scaled[temporal_cols] = scaler.transform(df_scaled[temporal_cols])

    for fold_idx in range(folds):
        print('------------------fold {}------------------'.format(fold_idx + 1))
        fold_label = 'fold-{}'.format(fold_idx + 1)
        save_dir = SAVE_DIR + fold_label + '/'
        os.makedirs(save_dir, exist_ok=True)
        if reward_label == 'rewards_90d':
            # Spilt dataset according to patient-level outcome
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3,
                                                                random_state=fold_idx + 1)
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, stratify=y_test, test_size=0.5,
                                                            random_state=fold_idx + 1)

            train_data = df_scaled[df_scaled['traj'].isin(X_train)].reset_index(drop=True)
            val_data = df_scaled[df_scaled['traj'].isin(X_val)].reset_index(drop=True)
            test_data = df_scaled[df_scaled['traj'].isin(X_test)].reset_index(drop=True)

            file_names = ['df_train', 'df_val', 'df_test']
            for idx, df in enumerate([train_data, val_data, test_data]):
                df.to_csv(save_dir + '/{}.csv'.format(file_names[idx]), index=False)

            tuple_names = ['train_set_tuples', 'val_set_tuples', 'test_set_tuples']
            for idx, df in enumerate([train_data, val_data, test_data]):
                states, acts, lengths, outcomes = get_tensor_data(df['traj'].unique(), df, outcome_col=reward_label)
                torch.save((states, acts, lengths, outcomes), os.path.join(save_dir, tuple_names[idx]))
                data_replay_buffer = prepare_replay_buffer(tensor_tuple=torch.load(os.path.join(save_dir, tuple_names[idx])), args=args)
                torch.save(data_replay_buffer, save_dir + '{}_replay_buffer'.format(tuple_names[idx]))
        elif reward_label == 'rewards_icu':
            test_data = df_scaled
            test_data.to_csv(save_dir + '/df_test.csv', index=False)

            tuple_names = ['test_set_tuples']
            for idx, df in enumerate([test_data]):
                states, acts, lengths, outcomes = get_tensor_data(df['traj'].unique(), df, outcome_col=reward_label)
                torch.save((states, acts, lengths, outcomes), os.path.join(save_dir, tuple_names[idx]))
                data_replay_buffer = prepare_replay_buffer(tensor_tuple=torch.load(os.path.join(save_dir, tuple_names[idx])), args=args)
                torch.save(data_replay_buffer, save_dir + '{}_replay_buffer'.format(tuple_names[idx]))
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Specify Params for MDP Preparation")
    # Refer from <Deep Reinforcement Learning for Sepsis Treatment>
    parser.add_argument('--C_0', type=float, default=-0.025)
    parser.add_argument('--C_1', type=float, default=-0.125)
    parser.add_argument('--C_2', type=float, default=-2.0)
    parser.add_argument('--C_3', type=float, default=0.05)
    parser.add_argument('--C_4', type=float, default=0.05)
    parser.add_argument('--terminal_coeff', type=int, default=15)
    args = parser.parse_args()
    rewards = ['rewards_90d', 'rewards_icu']
    for reward_label in rewards:
        split_sepsis_cohort(reward_label=reward_label)
