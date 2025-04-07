# -*- coding: utf-8 -*-
"""
@Time ： 2025/3/30 20:22
@Auth ： Hongwei
@File ：plot_for_nature_portfolio.py
@IDE ：PyCharm
"""

from definitions import *
from src.visualization.plot_surv_rate_with_expected_return import Q_cut_stat_surv_rate

color_list = [
    '#3e3e3e',
    # heuristic
    '#913175',
    '#560764',
    # FQI-based
    '#fc624d',
    '#fca7a7',
    # DQN-based
    '#f7c9c0',
    '#d59aad',
    '#a26085',
    '#a06275',
    '#af95b0',
    '#7249d6',
    # My model
    '#000249',
]

marker_list = ['o', '1', '2', 's', 'p', 'D', 'h', 'x', '^', 'H', '*', 'P']


def plot_boxplot(res_dict, fig, dataset_label):
    # xlabel上标记model name，不单列legend
    subplot_coff = 0 if dataset_label == 'rewards_90d' else 2
    dataset_name = ' on MIMIC-IV' if dataset_label == 'rewards_90d' else ' on eICU'
    for i, metric in enumerate(['WIS', 'WDR']):
        ax = fig.add_subplot(4, 2, i + subplot_coff + 1)
        bps = []
        for j, model in enumerate(models):
            eval_res = np.array(res_dict[model][metric])
            model_color = color_list[j]
            bp = ax.boxplot(eval_res, positions=[j], widths=0.6,
                            boxprops={'color': model_color, 'facecolor': model_color, 'alpha': 0.6},
                            medianprops={'color': 'blue'},
                            whiskerprops={'color': model_color}, capprops={'color': model_color},
                            flierprops={'markerfacecolor': model_color, 'markeredgecolor': model_color,
                                        'markersize': 2},
                            showmeans=True,
                            meanprops={'marker': '*', 'markerfacecolor': 'red', 'markeredgecolor': 'red',
                                       'markersize': 5},
                            patch_artist=True)
            bps.append(bp)
        ax.axvline(x=0.5, linestyle='--', color='black', alpha=0.3)
        ax.axvline(x=2.5, linestyle='--', color='black', alpha=0.3)
        ax.axvline(x=4.5, linestyle='--', color='black', alpha=0.3)
        ax.axvline(x=8.5, linestyle='--', color='black', alpha=0.3)
        ax.axvline(x=10.5, linestyle='--', color='black', alpha=0.3)
        ax.grid(axis='y', linestyle='dashed')
        ax.set_xticks(range(len(models)), model_names, fontsize=8, rotation=55)
        ax.set_xlabel('({}) PH'.format(chr(ord('a') + i + subplot_coff)) + metric + dataset_name, labelpad=8, fontsize=12)
        if metric == 'WIS':
            ax.set_ylabel('Values', fontsize=12)
        ylim = [-4, 13] if dataset_label == 'rewards_90d' else [2.5, 13]
        ax.set_ylim(ylim)


def plot_surv_rates(surv_rates, fig):
    interval_left = -15
    interval_right = 13
    threshold = np.arange(interval_left, interval_right, 1).tolist()
    plot_array = np.vstack((np.nanmean(surv_rates, axis=0),
                            np.nanmean(surv_rates, axis=0)+2*np.nanstd(surv_rates, axis=0),
                            np.nanmean(surv_rates, axis=0)-2*np.nanstd(surv_rates, axis=0)))
    plot_array[plot_array < 0] = 0
    plot_array[plot_array > 1] = 1
    x_list = [i + 0.5 for i in range(len(threshold) - 1)]

    ax = fig.add_subplot(4, 2, 5)
    mycolor = '#02B5A6'
    ax.plot(np.array(x_list), plot_array[0, :], color=mycolor, label='Mean Survival Rate')
    ax.fill_between(np.array(x_list), plot_array[1, :], plot_array[2, :], color=mycolor, alpha=0.25, label='95% Confidence Interval')

    mimic_color = '#084081'
    ax.axvline(x=20.5, linestyle='-', color=mimic_color, alpha=0.5)
    ax.axhline(y=0.69, linestyle='-', color=mimic_color, alpha=0.5)
    ax.scatter(20.5, 0.69, marker='p', c=mimic_color, label='Clinician Policy in MIMIC-IV')
    ax.text(20.8, 0.63, '72.54%', color=mimic_color, fontsize=10)

    eicu_color = '#913175'
    ax.axvline(x=23, linestyle='-', color=eicu_color, alpha=0.5)
    ax.axhline(y=0.815, linestyle='-', color=eicu_color, alpha=0.5)
    ax.scatter(23, 0.815, marker='^', c=eicu_color, label='Clinician Policy in eICU')
    ax.text(23.3, 0.76, '83.05%', color=eicu_color, fontsize=10)

    ax.set_xticks(np.arange(0, len(threshold), 3), np.arange(interval_left, interval_right, 3), fontsize=8)
    ax.set_xlabel('Expected Return\n({})'.format(chr(ord('a')+4)), fontsize=12)
    ax.set_ylabel('Survival Rate', fontsize=12)
    ax.set_ylim([0, 1])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='both', linestyle='dashed')


def analyse_dosage_diff_for_myModel(folds_plot_dict, fig):
    mortalitys = []
    for fold in list(folds_plot_dict.keys()):
        plot_dict = folds_plot_dict[fold]['HOPAS-B']
        mortality = np.array(plot_dict['mortality'])
        mortalitys.append(mortality.tolist())

    X_label, Y = np.floor(plot_dict['x_bins']), np.array(mortalitys)
    ax = fig.add_subplot(4, 2, 6)
    Y_mean, Y_std = np.nanmean(Y, axis=0), np.nanstd(Y, axis=0)
    Y_min = np.min(Y_mean[Y_mean != 0])

    X = [i for i in list(range(len(Y_mean)))]
    mycolor = '#fca7a7'
    ax.plot(X, Y_mean, color=mycolor, alpha=1, label='Mean Mortality')

    ax.fill_between(X, Y_mean + 2 * Y_std, Y_mean - 2 * Y_std, color=mycolor, alpha=0.5, label='95% Confidence Interval')
    ax.axvline(x=len(X_label) / 2 - 0.5, color='teal', linestyle='-', alpha=0.5, label='Same Dosage with HOPAS')
    ax.axhline(y=Y_min, color='blue', linestyle='-', alpha=0.5, label='Minimum Mortality')
    ax.set_xticks(range(len(X_label)), X_label, fontsize=8, rotation=-40)
    ax.set_xlabel('Average Dose Excess per Patient\n({})'.format(chr(ord('a')+5)), fontsize=12)
    ax.set_ylabel('Mortality', fontsize=12)
    ax.text(-0.5, Y_min + 0.01, '{:.2f}'.format(Y_min), fontsize=10, color='blue')
    ax.set_ylim(0, 0.8)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', linestyle='dashed')


def plot_action_pdf(fig, all_data, cluster_0, cluster_1, cluster_2, subplot_idx):
    ax = fig.add_subplot(4, 2, subplot_idx)
    fontsize = 12
    # heatmap plot
    action_matrix = []
    for i, action_values in enumerate([all_data, cluster_0, cluster_1, cluster_2]):
        density_values = action_values.value_counts().sort_index().values
        action_matrix.append((density_values / density_values.sum()).tolist())
    action_matrix = np.array(action_matrix)

    from matplotlib.colors import LinearSegmentedColormap
    my_cmap = LinearSegmentedColormap.from_list('my_cmap', ['#e1dad6', '#ea9c9d'])  # #915afe  #e3d0ff  '#f7d5ce', '#a26f8f'  #ea9c9d

    heatmap = ax.imshow(action_matrix, cmap=my_cmap, interpolation='nearest')
    cbar = plt.colorbar(heatmap, ax=ax, shrink=0.35, aspect=8, pad=0.08)
    cbar.ax.tick_params(labelsize=8)
    ax.text(4.5, 0.3, "Probability", fontsize=fontsize)
    for i in range(action_matrix.shape[0]):
        for j in range(action_matrix.shape[1]):
            ax.text(j, i, '{:.4f}'.format(action_matrix[i, j]), ha='center', va='center', color='black', fontsize=fontsize)
    # ax.set_xlabel('Action', fontsize=fontsize)
    ax.set_xticks(range(action_dim), ['0', '1', '2', '3', '4'], fontsize=fontsize)
    ax.set_yticks(range(4), ['All data', 'Cluster 0', 'Cluster 1', 'Cluster 2'], fontsize=fontsize)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    if subplot_idx == 7:
        title_string = 'Heparin administrated by clinicians'
    elif subplot_idx == 8:
        title_string = 'Heparin recommended by HOPAS'
    ax.set_xlabel('Action\n({}) {}'.format(chr(ord('a')+subplot_idx-1), title_string), fontsize=12)
    # ax.set_title('({}) {}'.format(chr(ord('a')+subplot_idx-1), title_string), fontsize=16, y=-0.35)
    pass


if __name__ == '__main__':
    # ***********************************Box plot***********************************
    models = [
        'Clinician',
        'Random', 'Zero drug',
        'RF-FQI', 'XGB-FQI',
        'DQN', 'DoubleDQN', 'DuelingDQN', 'DuelingDoubleDQN', 'DQN_CQL', 'WD3QNE',
        'HOPAS-B'
    ]
    model_names = [
        'Clinician',
        'Random', 'Zero drug',
        'RF-FQI', 'XGB-FQI',
        'DQN', 'DoubleDQN', 'DuelingDQN', 'DuelingDoubleDQN', 'CQL', 'WD3QNE',
        'HOPAS'
    ]
    fig = plt.figure(figsize=(12, 16), dpi=330)
    for data_idx, dataset in enumerate(['rewards_90d', 'rewards_icu']):
        pickle_file = load_pickle(RESULT_DIR + '{}_test-eval_res.pkl'.format(dataset))
        res_dict = {}
        for i, model_name in enumerate(models):
            res_dict[model_name] = {}
            for j, metric in enumerate(['WIS', 'WDR']):
                if model_name == 'Clinician':
                    # Soc的eval_res可以从任意一个模型的dict中取
                    metric_values = [pickle_file['DQN'][fold_label]['Clin {}'.format(eval_policy_type)][metric]
                                     for fold_label in list((pickle_file['DQN'].keys()))]
                else:
                    metric_values = [pickle_file[model_name][fold_label]['Agent {}'.format(eval_policy_type)][metric]
                                     for fold_label in list((pickle_file[model_name].keys()))]
                res_dict[model_name][metric] = metric_values
        plot_boxplot(res_dict, fig, dataset)

    # ***********************************Survival rate***********************************
    surv_rates = []
    val_Q_with_outcome = load_pickle(RESULT_DIR + '/val-Q_with_outcomes.pkl')
    test_Q_with_outcome = load_pickle(RESULT_DIR + '/test-Q_with_outcomes.pkl')
    for fold_idx in tqdm(range(folds)):
        fold_label = 'fold-{}'.format(fold_idx + 1)
        Q_with_outcome = pd.concat([val_Q_with_outcome[fold_label], test_Q_with_outcome[fold_label]])
        surv_rates.append(Q_cut_stat_surv_rate(Q_with_outcome))
    plot_surv_rates(np.array(surv_rates), fig)

    # ***********************************Dosage diff***********************************
    folds_plot_dict = load_pickle(RESULT_DIR + 'new_dosage_diff_with_mortality.pkl')
    analyse_dosage_diff_for_myModel(folds_plot_dict, fig)

    # ***********************************Action heatmap***********************************
    # Note: action of clinicians
    df_with_labels = pd.read_csv(DATA_DIR + 'all_patient_with_group_label.csv')
    all_patient_with_discretized_action = pd.read_csv(DATA_DIR + '/mimic-iv-sepsis_RAW_withTimes.csv')
    df_with_labels['action'] = all_patient_with_discretized_action['action']
    cluster_0 = df_with_labels[df_with_labels['group_label'] == 0].reset_index(drop=True)
    cluster_1 = df_with_labels[df_with_labels['group_label'] == 1].reset_index(drop=True)
    cluster_2 = df_with_labels[df_with_labels['group_label'] == 2].reset_index(drop=True)
    plot_action_pdf(fig, df_with_labels['action'], cluster_0['action'], cluster_1['action'], cluster_2['action'], 7)

    # Note: action of HOPAS-B
    concat_recommended_action_with_group_label = load_pickle(RESULT_DIR + 'concat_recommended_action_with_group_label.pkl')
    all_data = pd.DataFrame(np.array(concat_recommended_action_with_group_label), columns=['recommended action', 'group_label'])
    cluster_0 = all_data[all_data['group_label'] == 0].reset_index(drop=True)
    cluster_1 = all_data[all_data['group_label'] == 1].reset_index(drop=True)
    cluster_2 = all_data[all_data['group_label'] == 2].reset_index(drop=True)
    plot_action_pdf(fig, all_data['recommended action'], cluster_0['recommended action'], cluster_1['recommended action'], cluster_2['recommended action'], 8)

    plt.tight_layout()
    plt.savefig(FIG_DIR + 'PolicyComparison.pdf')
    plt.show()
    pass
