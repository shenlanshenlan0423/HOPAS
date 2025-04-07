# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/21 15:08
@Auth ： Hongwei
@File ：table_present.py
@IDE ：PyCharm
"""
from definitions import *
from src.eval.agent_eval import run_evaluate
from src.main import parse_args, set_seed
from src.utils.utils import paired_t_test, star_by_p_value


def run_other_benchmark(args):
    # For generating pickle file, run only once
    pickle_file = load_pickle(RESULT_DIR + '/{}_test-eval_res.pkl'.format(args.outcome_label))
    pickle_file['Random'] = {}
    pickle_file['Zero drug'] = {}

    for idx in tqdm(range(folds)):
        data_path = DATA_DIR + '/{}/fold-{}/'.format(args.outcome_label, idx + 1)
        fold_label = 'fold-{}'.format(idx + 1)
        behavior_policy = load_pickle(RESULT_DIR + '/{}-test-bps.pkl'.format(args.outcome_label))[fold_label]

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
        fqi_model = load_pickle(MODELS_DIR + '/[{}]-RF-FQI.pkl'.format(fold_label))

        random_policy = np.random.rand(states.shape[0], action_dim)
        pickle_file['Random'][fold_label] = run_evaluate(random_policy / random_policy.sum(axis=1, keepdims=True),
                                                         transition_dict, behavior_policy, None,
                                                         fqi_model, args)

        zero_drug_policy = np.zeros_like(random_policy)  # 完全不用药 action均为0
        zero_drug_policy[:, 0] = 1
        pickle_file['Zero drug'][fold_label] = run_evaluate(zero_drug_policy, transition_dict, behavior_policy,
                                                            None, fqi_model, args)

    save_pickle(pickle_file, RESULT_DIR + '/{}_test-eval_res.pkl'.format(args.outcome_label))


def main_table(pickle_file):
    if args.outcome_label == 'rewards_90d':
        models = [
            'Clinician',
            'Random', 'Zero drug',
            'RF-FQI', 'XGB-FQI',
            'DQN', 'DoubleDQN', 'DuelingDQN', 'DuelingDoubleDQN', 'DQN_CQL', 'WD3QNE',
            'HOPAS-A', 'HOPAS-B', 'HOPAS_wo_seq2seq-B'
        ]
    elif args.outcome_label == 'rewards_icu':
        models = [
            'Clinician',
            'Random', 'Zero drug',
            'RF-FQI', 'XGB-FQI',
            'DQN', 'DoubleDQN', 'DuelingDQN', 'DuelingDoubleDQN', 'DQN_CQL', 'WD3QNE',
            'HOPAS-B'
        ]
    result_dataframe = pd.DataFrame(columns=['Policy', 'PHWIS', 'PHWIS Gain', 'PHWDR', 'PHWDR Gain', 'ESR (%)', 'ESR Gain'])
    result_dataframe['Policy'] = models
    WDRs_for_ESR = {}
    for i, model_name in enumerate(models):
        for j, metric in enumerate(['WIS', 'WDR']):
            proposed_model_values = [pickle_file['HOPAS-B'][fold_label]['Agent {}'.format(eval_policy_type)][metric]
                                     for fold_label in list((pickle_file['HOPAS-B'].keys()))]
            if model_name == 'Clinician':
                # Soc的eval_res可以从任意一个模型的dict中取
                metric_values = [pickle_file['DQN'][fold_label]['Clin {}'.format(eval_policy_type)][metric]
                                 for fold_label in list((pickle_file['DQN'].keys()))]
            else:
                metric_values = [pickle_file[model_name][fold_label]['Agent {}'.format(eval_policy_type)][metric]
                                 for fold_label in list((pickle_file[model_name].keys()))]
            if metric == 'WDR':
                WDRs_for_ESR[model_name] = metric_values
            t_statistic, p_value = paired_t_test(proposed_model_values, metric_values)
            String = star_by_p_value(p_value, '{:.2f} (±{:.2f})'.format(np.mean(metric_values), np.std(metric_values)))
            result_dataframe.iloc[i, j * 2 + 1] = String

    Mean_WIS = [float(i.split(' ')[0]) for i in result_dataframe['PHWIS'].tolist()]
    Mean_WDR = [float(i.split(' ')[0]) for i in result_dataframe['PHWDR'].tolist()]

    if args.outcome_label == 'rewards_90d':
        # Note: MIMIC-IV中有消融实验的效果展示,所以HOPAS的索引是-2
        result_dataframe['PHWIS Gain'] = ['{:.2f}%'.format((Mean_WIS[-2] - i) / i * 100) for i in Mean_WIS]
        result_dataframe['PHWDR Gain'] = ['{:.2f}%'.format((Mean_WDR[-2] - i) / i * 100) for i in Mean_WDR]
        save_pickle(WDRs_for_ESR, RESULT_DIR + 'mimic_WDRs_for_ESR.pkl')
        final_estimated_surv_rate = load_pickle(RESULT_DIR + 'mimic_ESR.pkl') * 100
        # paired-t-test for ESR metric
        proposed_model_values = final_estimated_surv_rate[-2].tolist()
        for idx in range(len(models)):
            metric_values = final_estimated_surv_rate[idx].tolist()
            t_statistic, p_value = paired_t_test(proposed_model_values, metric_values)
            String = star_by_p_value(p_value, '{:.2f} (±{:.2f})'.format(np.mean(metric_values), np.std(metric_values)))
            result_dataframe['ESR (%)'].iloc[idx] = String
        mean_surv_rate = [float(i.split(' ')[0]) for i in result_dataframe['ESR (%)'].tolist()]
        result_dataframe['ESR Gain'] = ['{:.2f}%'.format((mean_surv_rate[-2] - i) / i * 100) for i in mean_surv_rate]
    elif args.outcome_label == 'rewards_icu':
        result_dataframe['PHWIS Gain'] = ['{:.2f}%'.format((Mean_WIS[-1] - i) / i * 100) for i in Mean_WIS]
        result_dataframe['PHWDR Gain'] = ['{:.2f}%'.format((Mean_WDR[-1] - i) / i * 100) for i in Mean_WDR]
        save_pickle(WDRs_for_ESR, RESULT_DIR + 'eicu_WDRs_for_ESR.pkl')
        final_estimated_surv_rate = load_pickle(RESULT_DIR + 'eicu_ESR.pkl') * 100
        # paired-t-test for ESR metric
        proposed_model_values = final_estimated_surv_rate[-1].tolist()
        for idx in range(len(models)):
            metric_values = final_estimated_surv_rate[idx].tolist()
            t_statistic, p_value = paired_t_test(proposed_model_values, metric_values)
            String = star_by_p_value(p_value, '{:.2f} (±{:.2f})'.format(np.mean(metric_values), np.std(metric_values)))
            result_dataframe['ESR (%)'].iloc[idx] = String
        mean_surv_rate = [float(i.split(' ')[0]) for i in result_dataframe['ESR (%)'].tolist()]
        result_dataframe['ESR Gain'] = ['{:.2f}%'.format((mean_surv_rate[-1] - i) / i * 100) for i in mean_surv_rate]

    print(result_dataframe.to_markdown(index=False))
    return result_dataframe


if __name__ == '__main__':
    args = parse_args()
    set_seed(seed=args.seed)
    run_other_benchmark(args)
    res_table = main_table(load_pickle(RESULT_DIR + '/{}_test-eval_res.pkl'.format(args.outcome_label)))
    res_table.to_excel(TABLE_DIR + '/{}_test_res.xlsx'.format(args.outcome_label), index=False)

    # -----------------------------------------------External Validation------------------------------------------------
    args.outcome_label = 'rewards_icu'
    run_other_benchmark(args)
    res_table = main_table(load_pickle(RESULT_DIR + '/{}_test-eval_res.pkl'.format(args.outcome_label)))
    res_table.to_excel(TABLE_DIR + '/{}_test_res.xlsx'.format(args.outcome_label), index=False)
