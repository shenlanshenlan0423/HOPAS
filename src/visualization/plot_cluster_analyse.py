# -*- coding: utf-8 -*-
"""
@Time ： 2024/12/18 9:30
@Auth ： Hongwei
@File ：plot_cluster_analyse.py
@IDE ：PyCharm
"""
from definitions import *
import umap

color_list = [
    '#4c66d6',  # blue
    '#e16751',  # red
    '#e1dad6',  # grey
]
mark_list = ['s', '^', 'P']
linestyle_list = ['solid', 'dashed', 'dashdot']


def plot_StratificationRes(X, cluster_model, all_data, cluster_0, cluster_1, cluster_2):
    fig = plt.figure(figsize=(14, 12), dpi=330)
    # umap_model = umap.UMAP(random_state=42)
    # save_pickle(umap_model.fit_transform(X), RESULT_DIR+'mapped_X_train.pkl')

    plot_elbow(fig, 1)
    plot_principal(fig, load_pickle(RESULT_DIR+'mapped_X_train.pkl'), cluster_model.predict(X), 2)
    plot_surv_dead_varying_class(fig, cluster_0, cluster_1, cluster_2, 3)
    plot_key_features_pattern(fig, cluster_0, cluster_1, cluster_2)
    cluster_counts = [len(i['traj'].unique().tolist()) for i in [cluster_0, cluster_1, cluster_2]]
    plot_SOFA_distribution(fig, cluster_0['SOFA'], cluster_1['SOFA'], cluster_2['SOFA'], cluster_counts)

    plt.tight_layout()
    plt.savefig(FIG_DIR + '/PhenotypesIdentificationResult.pdf')
    plt.show()
    pass


def plot_elbow(fig, subplot_idx):
    eval_res = load_pickle(RESULT_DIR + 'cluster_eval_res_for_elbow.pkl')
    metric_values = []
    for n_clusters in list(eval_res.keys()):
        metric_value = eval_res[n_clusters]['Silhouette coefficient']
        metric_values.append(metric_value)

    ax = fig.add_subplot(3, 3, subplot_idx)
    fontsize = 12
    x_list = list(eval_res.keys())
    ax.plot(x_list, metric_values, color='#fca7a7', alpha=0.5)
    ax.scatter(x_list, metric_values, color='#e67d80', marker='.', s=36)
    ax.scatter(3, 0.1446156, color='red', marker='*', s=36)
    ax.text(3.15, 0.15, 'Best K', color='red', fontsize=15)
    ax.set_xticks(range(2, 10))
    ax.set_yticks(np.arange(0.08, 0.3, 0.04))
    ax.set_xlabel('Number of Clusters Initialized\n(a)', fontsize=fontsize)
    ax.set_ylabel('Silhouette Coefficient', fontsize=fontsize)
    plt.grid(linestyle='dashed')


def plot_principal(fig, X, labels, subplot_idx):
    ax = fig.add_subplot(3, 3, subplot_idx)
    fontsize = 12
    for i, label in enumerate(sorted(pd.Series(labels).unique().tolist())):
        subclass_X = X[np.where(labels == label)]
        ax.scatter(subclass_X[:, 0], subclass_X[:, 1], c=color_list[i], marker=mark_list[i], s=24, alpha=0.45)
    ax.scatter(1, 5, c=color_list[0], marker=mark_list[0], s=80, label='Cluster 0')
    ax.scatter(9, 5, c=color_list[1], marker=mark_list[1], s=80, label='Cluster 1')
    ax.scatter(3, 7, c=color_list[2], marker=mark_list[2], s=80, label='Cluster 2')
    ax.set_xlabel('UMAP 1 of Patient Representation\n(b)', fontsize=fontsize)
    ax.set_ylabel('UMAP 2 of Patient Representation', fontsize=fontsize)
    ax.legend(loc='lower right', fontsize=fontsize)
    ax.grid(axis='both', linestyle='dashed')


def plot_surv_dead_varying_class(fig, cluster_0, cluster_1, cluster_2, subplot_idx):
    class_mortality = []
    for cluster in [cluster_0, cluster_1, cluster_2]:
        class_mortality.append(cluster.groupby('traj')['90d mortality'].mean().sum() / len(cluster['traj'].unique().tolist()))
    pass
    surv_dead_varying_class = np.array([(1 - np.array(class_mortality)).tolist(), class_mortality])
    ax = fig.add_subplot(3, 3, subplot_idx)

    fontsize = 12
    bar_height = 0.3
    y = range(3)
    bars = ax.barh([yi + bar_height/2 for yi in y], surv_dead_varying_class[0, :], height=bar_height, color='#b3e8e3', label='Survival Rate')

    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2, '{:.2f}'.format(width), ha='left', va='center')
    bars = ax.barh([yi - bar_height/2 for yi in y], surv_dead_varying_class[1, :], height=bar_height, color='#f39ca2', label='Mortality')

    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2, '{:.2f}'.format(width), ha='left', va='center')

    ax.set_xlabel('Probability\n(c)', fontsize=fontsize)
    ax.set_xlim([0, 1])
    ax.set_yticks(y, ['Cluster 0', 'Cluster 1', 'Cluster 2'])
    ax.legend(loc='center right', fontsize=8)
    ax.grid(axis='both', linestyle='dashed')


def plot_key_features_pattern(fig, cluster_0, cluster_1, cluster_2):
    color_list = [
        '#122794',  # blue
        '#f44f38',  # red
        '#000000',  # grey
    ]
    features = ['SOFA', 'Platelets', 'Lactate']
    X = np.arange(0, horizon, 1)
    for idx, feature in enumerate(features):
        ax = fig.add_subplot(3, 3, idx + 4)
        for cluster_idx, cluster_data in enumerate([cluster_0, cluster_1, cluster_2]):
            gps = cluster_data.groupby('traj')
            SOFA_values, Platelets_values, Lactate_values = [], [], []
            for traj, gp in gps:
                SOFA_values.append(gp['SOFA'].values.tolist() + [np.nan] * (horizon - gp.shape[0]))
                Platelets_values.append(gp['Platelets'].values.tolist() + [np.nan] * (horizon - gp.shape[0]))
                Lactate_values.append(gp['Lactate'].values.tolist() + [np.nan] * (horizon - gp.shape[0]))
            SOFA_values, Platelets_values, Lactate_values= np.array(SOFA_values), np.array(Platelets_values), np.array(Lactate_values)
            SOFA_mean, SOFA_std = np.nanmean(SOFA_values, axis=0), np.nanstd(SOFA_values, axis=0)
            Platelets_mean, Platelets_std = np.nanmean(Platelets_values, axis=0), np.nanstd(Platelets_values, axis=0)
            Lactate_mean, Lactate_std = np.nanmean(Lactate_values, axis=0), np.nanstd(Lactate_values, axis=0)
            if cluster_idx == 0:
                # 轨迹长度为14的样本太少，得到的结果不具备代表性
                SOFA_mean[13:] = np.nan
                Platelets_mean[13:] = np.nan
                Lactate_mean[13:] = np.nan
            if idx == 0:
                ax.scatter(X, SOFA_mean, color=color_list[cluster_idx], alpha=0.8, marker=mark_list[cluster_idx], s=18)
                ax.plot(X, SOFA_mean, color=color_list[cluster_idx], linestyle=linestyle_list[cluster_idx], linewidth=1.5, alpha=0.8, label='Cluster {}'.format(cluster_idx))
                ax.set_ylim([0, 10])
                ax.set_xlabel('Timestamp\n(d)', fontsize=12)
                ax.legend(loc='lower right', fontsize=12)
            elif idx == 1:
                ax.scatter(X, Platelets_mean, color=color_list[cluster_idx], alpha=0.8, marker=mark_list[cluster_idx], s=18)
                ax.plot(X, Platelets_mean, color=color_list[cluster_idx], alpha=0.8, linestyle=linestyle_list[cluster_idx], linewidth=1.5, label='Cluster {}'.format(cluster_idx))
                ax.set_ylim([100, 155])
                ax.set_xlabel('Timestamp\n(e)', fontsize=12)
                ax.legend(loc='upper right', fontsize=12)
            elif idx == 2:
                ax.scatter(X, Lactate_mean, color=color_list[cluster_idx], alpha=0.8, marker=mark_list[cluster_idx], s=18)
                ax.plot(X, Lactate_mean, color=color_list[cluster_idx], alpha=0.8, linestyle=linestyle_list[cluster_idx], linewidth=1.5, label='Cluster {}'.format(cluster_idx))
                ax.set_ylim([1.5, 4])
                ax.set_xlabel('Timestamp\n(f)', fontsize=12)
                ax.legend(loc='upper right', fontsize=12)
        ax.set_ylabel(f'{feature}', fontsize=12)
        ax.set_xticks(np.arange(0, horizon, 3), np.arange(0, horizon, 3))
        plt.grid(axis='both', linestyle='solid')
    pass


def plot_SOFA_distribution(fig, cluster_0, cluster_1, cluster_2, cluster_counts):
    X = np.arange(0, 25, 1)
    for i, SOFA in enumerate([cluster_0, cluster_1, cluster_2]):
        ax = fig.add_subplot(3, 3, i + 7)
        SOFA_values = SOFA.round().value_counts()
        if SOFA_values.shape[0] != X.shape[0]:
            nan_index = [i for i in X if i not in SOFA_values.index]
            for j in nan_index:
                SOFA_values.loc[j] = 0
        SOFA_values = SOFA_values.sort_index()
        # 有的classSOFA不全，需要补充
        Y = SOFA_values.values / SOFA_values.sum()
        ax.bar(X, Y, color='#a68ae6', alpha=0.8)
        ax.fill_between([-0.4, 5], 0, 0.35, color='#8adacf', alpha=0.15, label='Low SOFA')
        ax.fill_between([5, 15], 0, 0.35, color='#eebb44', alpha=0.15, label='Mid SOFA')
        ax.fill_between([15, 24], 0, 0.35, color='red', alpha=0.15, label='High SOFA')
        ax.set_xlim([-1, 23.5])
        ax.set_xlabel('Cluster {} (n={})\n({})'.format(i, cluster_counts[i], chr(ord('a') + i + 6)), fontsize=12)
        ax.set_xticks(range(0, 25, 3), range(0, 25, 3))
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_ylim([0, 0.32])
        ax.legend(loc="upper right", prop={'size': 12})
        ax.grid(axis='x', linestyle='dashed')
    pass


if __name__ == '__main__':
    df_with_labels = pd.read_csv(DATA_DIR + 'all_patient_with_group_label.csv')
    all_patient_with_discretized_action = pd.read_csv(DATA_DIR + '/mimic-iv-sepsis_RAW_withTimes.csv')
    df_with_labels['action'] = all_patient_with_discretized_action['action']
    cluster_0 = df_with_labels[df_with_labels['group_label'] == 0].reset_index(drop=True)
    cluster_1 = df_with_labels[df_with_labels['group_label'] == 1].reset_index(drop=True)
    cluster_2 = df_with_labels[df_with_labels['group_label'] == 2].reset_index(drop=True)

    patient_level_representations, kmeans = load_pickle(RESULT_DIR + 'res_for_plot_Stratification.pkl')
    plot_StratificationRes(patient_level_representations, kmeans, df_with_labels['action'], cluster_0, cluster_1, cluster_2)
    pass
