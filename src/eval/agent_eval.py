# -*- coding: utf-8 -*-
"""
@Auth ： Hongwei
@File ：agent_eval.py
@IDE ：PyCharm
"""
from definitions import *
from sklearn.neighbors import NearestNeighbors
from src.utils.utils import Physiological_distance_kernel


def KNN_approx_behavior_policy_for_test_data(transition_dict_for_train, transition_dict_for_test, n_neighbors=300):
    train_states = torch.cat(transition_dict_for_train['states'], dim=0).cpu().numpy()
    train_actions = torch.cat(transition_dict_for_train['actions'], dim=0).cpu().numpy()
    test_states = torch.cat(transition_dict_for_test['states'], dim=0).cpu().numpy()

    train_states = Physiological_distance_kernel(train_states)
    test_states = Physiological_distance_kernel(test_states)

    # Clinicians typically make decisions based on their experience treating similar patients
    # Intuitively, states that are physiologically similar should
    # have similar treatment (behaviour policy) distributions.
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(train_states)
    distances, indices = neigh.kneighbors(test_states)
    neighbors_actions = train_actions[indices]
    actions_probs = (neighbors_actions.sum(axis=1) / n_neighbors)
    return actions_probs


def FQI_for_Q_estimate(transition_dict_for_train, args):
    states = torch.cat(transition_dict_for_train['states'], dim=0).cpu().numpy()
    actions = torch.cat(transition_dict_for_train['actions'], dim=0).cpu().numpy()
    rewards = torch.cat(transition_dict_for_train['rewards'], dim=0).cpu().numpy()
    next_states = torch.cat(transition_dict_for_train['next_states'], dim=0).cpu().numpy()

    max_q_values = np.zeros((states.shape[0]))  # Initialize Q
    i = np.hstack((states, actions))  # True state-action pair in training set
    action_space = pd.get_dummies(pd.Series(list(range(action_dim)))).values  # 所有可选动作 one-hot encoding
    i_t1 = np.hstack((np.repeat(next_states, action_dim, axis=0), np.tile(action_space, next_states.shape[0]).T))
    # <Transatlantic transferability of a new reinforcement learning model for optimizing haemodynamic treatment for critically ill patients with sepsis>
    # https://doi.org/10.1016/j.artmed.2020.102003  RF using 80 trees over 100 iterations
    model = RandomForestRegressor(n_estimators=80, max_depth=6, random_state=42)
    reward_mask = np.abs(rewards) < 12  # Note:  terminal state的max_q_values没有意义（参考自DQN的Q更新公式）；否则reward上限的持续增加会影响训练的数值稳定性
    loss_res = []
    with tqdm(total=args.max_iteration) as pbar:
        for _ in range(args.max_iteration):
            o = rewards + args.gamma * (max_q_values * reward_mask)
            sample_idx = np.random.choice(range(i.shape[0]), size=50000, replace=False)
            model.fit(X=i[sample_idx], y=o[sample_idx])  # Sampling for Acceleration
            q_values_hat = model.predict(i_t1).reshape(next_states.shape[0], action_dim)  # 估计next states在所有可选动作下的Q values
            mse_loss = np.mean((max_q_values - q_values_hat.max(axis=1)) ** 2)
            max_q_values = q_values_hat.max(axis=1)
            pbar.set_postfix({
                'Loss': '{}'.format(mse_loss)
            })
            pbar.update(1)
            loss_res.append(mse_loss)
    return model, loss_res


def get_Q_values_with_outcome(transition_dict, fqi_model):
    states = torch.cat(transition_dict['states'], dim=0)
    actions = torch.cat(transition_dict['actions'], dim=0).cpu().numpy()
    rewards = torch.cat(transition_dict['rewards'], dim=0).cpu().numpy()
    dones = torch.cat(transition_dict['dones'], dim=0).cpu().numpy()

    Q_s_a = fqi_model.predict(np.hstack((states.cpu().numpy(), actions)))
    rewards[dones == 0] = np.nan
    rewards[rewards > 0] = 0
    rewards[rewards < 0] = 1
    Q_with_outcome = np.vstack((Q_s_a, rewards)).T
    df = pd.DataFrame(Q_with_outcome, columns=['Q_s_a', 'outcome']).fillna(method='bfill')
    return df


def run_evaluate(Agent_stochastic_policy, transition_dict, behavior_policy, Q_estimate, fqi_model, args,
                 fold_label=None):
    states = torch.cat(transition_dict['states'], dim=0)
    actions = torch.cat(transition_dict['actions'], dim=0).cpu().numpy()
    rewards = torch.cat(transition_dict['rewards'], dim=0).cpu().numpy()
    dones = torch.cat(transition_dict['dones'], dim=0).cpu().numpy()

    Clinician_stochastic_policy = behavior_policy
    Clin_deterministic_policy = np.eye(action_dim)[Clinician_stochastic_policy.argmax(1)]

    # Note: ----------------------------------------For PH-WDR of Clin-----------------------------------------
    # State-action pair value
    Q_s_a = fqi_model.predict(np.hstack((states.cpu().numpy(), actions)))
    # For calculating state value
    action_space = pd.get_dummies(pd.Series(list(range(action_dim)))).values
    i_t1 = np.hstack((np.repeat(states.cpu().numpy(), action_dim, axis=0),
                      np.tile(action_space, states.shape[0]).T))  # 估计next states在所有可选动作下的Q values
    Q_s_all_a = fqi_model.predict(i_t1).reshape(states.shape[0], action_dim)

    if args.plot_flag and fold_label == 'fold-1':
        Deterministic_policys = load_pickle(RESULT_DIR + 'Deterministic_policys_for_plot.pickle')
        Deterministic_policys['Soc'] = [actions.argmax(axis=1).tolist(), 1]  # [policy, MCR]
        Deterministic_policys[args.model_name] = [Agent_stochastic_policy.argmax(axis=1).tolist(),
                                                  compute_MCR(Agent_stochastic_policy.argmax(axis=1),
                                                              actions.argmax(axis=1))]
        save_pickle(Deterministic_policys, RESULT_DIR + 'Deterministic_policys_for_plot.pickle')
        print('Save res for action matrix.')

    Agent_deterministic_policy = np.eye(action_dim)[Agent_stochastic_policy.argmax(1)]
    action_idx = Clin_deterministic_policy.argmax(1)  # Note: eval policy和behavior policy选择数据中某个action的概率
    policys_and_res = {
        'Agent Stochastic': [], 'Agent Deterministic': [],
        'Clin Stochastic': [], 'Clin Deterministic': [],
        'Data Stochastic action': [],
        'FQI Q_s_a': Q_s_a,
        'Agent V_s': (Agent_stochastic_policy * Q_s_all_a).sum(axis=1),
        'Clin V_s': (Clinician_stochastic_policy * Q_s_all_a).sum(axis=1),
        'Agent action': Agent_stochastic_policy.argmax(axis=1),
        'True action': action_idx,
    }
    for i in range(states.shape[0]):
        policys_and_res['Agent Stochastic'].append(Agent_stochastic_policy[i][action_idx[i]])
        policys_and_res['Agent Deterministic'].append(Agent_deterministic_policy[i][action_idx[i]])
        policys_and_res['Clin Stochastic'].append(Clinician_stochastic_policy[i][action_idx[i]])
        policys_and_res['Clin Deterministic'].append(Clin_deterministic_policy[i][action_idx[i]])
    policys_and_res['Data Stochastic action'] = policys_and_res['Clin Stochastic']
    policys_and_res['Data Deterministic action'] = policys_and_res['Clin Deterministic']
    result_dict = {}
    models = ['Agent {}'.format(eval_policy_type), 'Clin {}'.format(eval_policy_type)]
    for model in models:
        result_dict[model] = evaluate(policys_and_res, model, rewards, dones, args.gamma)
    return result_dict


def evaluate(policys_and_res, model, rewards, dones, gamma):
    if model[:4] == 'Clin':
        WIS, WDR = off_policy_evaluation(policys_and_res[model],
                                         policys_and_res['Data {} action'.format(eval_policy_type)],
                                         policys_and_res['FQI Q_s_a'],
                                         policys_and_res['Clin V_s'],
                                         rewards, dones, gamma)
        res_dict = {
            'WIS': WIS,
            'WDR': WDR,
            'MCR': 1
        }
    else:
        WIS, WDR = off_policy_evaluation(policys_and_res[model],
                                         policys_and_res['Data {} action'.format(eval_policy_type)],
                                         policys_and_res['FQI Q_s_a'],  # BUG: why
                                         policys_and_res['Agent V_s'],
                                         rewards, dones, gamma)
        res_dict = {
            'WIS': WIS,
            'WDR': WDR,
            'MCR': compute_MCR(policys_and_res['Agent action'], policys_and_res['True action'])
        }
    return res_dict


def print_results(evaluation_results, model_name):
    metrics = ['WIS', 'WDR', 'MCR']
    for metric in metrics:
        metric_values = [evaluation_results[fold_label]['Clin {}'.format(eval_policy_type)][metric]
                         for fold_label in list((evaluation_results.keys()))]
        print('Res. of Clin {}: {:.2f} (±{:.2f})'.format(metric, np.mean(metric_values), np.std(metric_values)))
        metric_values = [evaluation_results[fold_label]['Agent {}'.format(eval_policy_type)][metric]
                         for fold_label in list((evaluation_results.keys()))]
        print('Res. of {} {}: {:.2f} (±{:.2f})'.format(model_name, metric, np.mean(metric_values), np.std(metric_values)))


def off_policy_evaluation(pi_rls, pi_clins, Q_s_as, V_ss, rewards, dones, gamma):
    # Note: 对稀疏奖励和稠密奖励都适用
    # Per-Horizon Weighted Importance Sampling (PHWIS) and Per-Horizon Weighted Doubly Robust (PHWDR)
    # in "Behaviour Policy Estimation in Off-Policy Policy Evaluation: Calibration Matters"
    patient_lengths = np.where(dones == 1)[0].tolist()
    num_patients = len(patient_lengths)

    rho_total_list, rho_array = [], np.full((num_patients, horizon), np.nan)
    Q_s_a_array, V_s_array = np.full_like(rho_array, np.nan), np.full_like(rho_array, np.nan)
    cumulative_reward_list, reward_array = [], np.full_like(rho_array, np.nan)
    for patient_idx in range(num_patients):
        if patient_idx == 0:
            pi_rl = pi_rls[0:patient_lengths[patient_idx] + 1]
            pi_clin = pi_clins[0:patient_lengths[patient_idx] + 1]
            reward = rewards[0:patient_lengths[patient_idx] + 1]
            Q_s_a = Q_s_as[0:patient_lengths[patient_idx] + 1]
            V_s = V_ss[0:patient_lengths[patient_idx] + 1]
        else:
            pi_rl = pi_rls[patient_lengths[patient_idx - 1] + 1:patient_lengths[patient_idx] + 1]
            pi_clin = pi_clins[patient_lengths[patient_idx - 1] + 1:patient_lengths[patient_idx] + 1]
            reward = rewards[patient_lengths[patient_idx - 1] + 1:patient_lengths[patient_idx] + 1]
            Q_s_a = Q_s_as[patient_lengths[patient_idx - 1] + 1:patient_lengths[patient_idx] + 1]
            V_s = V_ss[patient_lengths[patient_idx - 1] + 1:patient_lengths[patient_idx] + 1]

        T = len(pi_clin)
        rho = 1
        cumulative_reward = 0
        for t in range(T):
            if pi_clin[t] == 0:
                # Note: Assumption in Page 5 of <Behaviour Policy Estimation in
                #  Off-Policy Policy Evaluation:  Calibration Matters>: if pi_clin[t] == 0 then pi_rl[t] == 0
                rho *= 0
            else:
                rho *= (pi_rl[t] / pi_clin[t])
            rho_array[patient_idx, t] = rho
            Q_s_a_array[patient_idx, t] = Q_s_a[t]
            V_s_array[patient_idx, t] = V_s[t]
            reward_array[patient_idx, t] = reward[t]
            cumulative_reward += gamma ** t * reward[t]
        rho_total_list.append(rho)
        cumulative_reward_list.append(cumulative_reward)
    WIS = compute_PHWIS(rho_array, cumulative_reward_list)
    WDR = compute_PHWDR(rho_array, reward_array, Q_s_a_array, V_s_array, gamma)
    return WIS, WDR


def compute_PHWIS(rho_array, cumulative_reward_list):
    traj_lengths = horizon - np.sum(np.isnan(rho_array), axis=1)
    # Concat step-wise rho, cumulative reward and trajectory length
    cols = ['traj_length'] + ['rho-{}'.format(i) for i in range(horizon)] + ['cumulative_reward']
    df_rho = pd.DataFrame(np.column_stack((traj_lengths, rho_array, np.array(cumulative_reward_list))), columns=cols)
    W = df_rho['traj_length'].value_counts().sort_index().values / rho_array.shape[0]  # Note: Outer weight
    patient_gps = df_rho.groupby('traj_length')
    group_V = []
    for traj_length, gp in patient_gps:
        last_timing_rho = gp['rho-{}'.format(int(traj_length - 1))].values
        cumulative_reward = gp['cumulative_reward'].values
        # 考虑当前时刻计算omega时分母为0的情况
        if last_timing_rho.sum() == 0:
            omega = np.full((gp.shape[0],), 1 / gp.shape[0])
        else:
            omega = last_timing_rho / last_timing_rho.sum()
        group_V.append(np.dot(omega, cumulative_reward))
    WIS = np.dot(W, np.array(group_V))
    return WIS


def compute_PHWDR(rho_array, reward_array, Q_s_a_array, V_s_array, gamma=0.99):
    traj_lengths = horizon - np.sum(np.isnan(rho_array), axis=1)
    cols = ['traj_length'] + ['rho-{}'.format(i) for i in range(horizon)] + \
           ['reward-{}'.format(i) for i in range(horizon)] + \
           ['Q_s_a-{}'.format(i) for i in range(horizon)] + ['V_s-{}'.format(i) for i in range(horizon)]
    df_rho = pd.DataFrame(np.column_stack((traj_lengths, rho_array, reward_array, Q_s_a_array, V_s_array)), columns=cols)
    W = df_rho['traj_length'].value_counts().sort_index().values / rho_array.shape[0]  # Note: Outer weight
    patient_gps = df_rho.groupby('traj_length')  # Grouping by trajectory length
    group_V = []
    # expected_returns = []
    for traj_length, gp in patient_gps:
        group_step_DR = []
        for t in range(int(traj_length)):
            # 考虑第0个时刻不存在t_minus_1_l的情况
            if t == 0:
                omega_t_minus_1_l = np.full((gp.shape[0],),
                                            1 / gp.shape[0])  # Note: 如果是第0个时刻，omega_t_minus_1_l为各个患者轨迹赋予相同的权重
            else:
                omega_t_minus_1_l = omega_t_l
            # 考虑当前时刻计算omega时分母为0的情况
            if gp['rho-{}'.format(t)].values.sum() == 0:
                omega_t_l = np.full((gp.shape[0],), 1 / gp.shape[0])
            else:
                omega_t_l = gp['rho-{}'.format(t)].values / gp['rho-{}'.format(t)].values.sum()

            item_1 = gamma ** t * omega_t_l * gp['reward-{}'.format(t)].values
            item_2 = gamma ** t * (
                    omega_t_l * gp['Q_s_a-{}'.format(t)].values - omega_t_minus_1_l * gp['V_s-{}'.format(t)].values)
            group_step_DR.append((item_1 - item_2).tolist())
        group_step_DR = np.array(group_step_DR).T
        group_DR = group_step_DR.sum(axis=1)
        group_V.append(group_DR.sum())
    WDR = np.dot(W, np.array(group_V))
    return WDR


def compute_MCR(pi_rls, pi_clins):
    match_list = []
    for _ in range(len(pi_rls)):
        if pi_rls[_] == pi_clins[_]:
            match_list.append(1)
        else:
            match_list.append(0)
    return match_list.count(1) / len(match_list)
