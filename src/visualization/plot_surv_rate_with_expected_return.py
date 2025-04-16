# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/17 10:52
@Auth ： Hongwei
@File ：plot_surv_rate_with_expected_return.py
@IDE ：PyCharm
"""
from definitions import *

interval_left = -15
interval_right = 13


def plot_surv_rates(surv_rates):
    threshold = np.arange(interval_left, interval_right, 1).tolist()
    plot_array = np.vstack((np.nanmean(surv_rates, axis=0),
                            np.nanmean(surv_rates, axis=0)+2*np.nanstd(surv_rates, axis=0),
                            np.nanmean(surv_rates, axis=0)-2*np.nanstd(surv_rates, axis=0)))
    plot_array[plot_array < 0] = 0
    plot_array[plot_array > 1] = 1
    x_list = [i + 0.5 for i in range(len(threshold) - 1)]

    fig = plt.figure(figsize=(6, 4), dpi=330)
    ax = fig.add_subplot(1, 1, 1)
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
    ax.set_xlabel('Expected Return', fontsize=12)
    ax.set_ylabel('Survival Rate', fontsize=12)
    ax.set_ylim([0, 1])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='both', linestyle='dashed')

    plt.tight_layout()
    plt.show()
    pass

    fit_array = np.vstack((np.arange(interval_left, interval_right-1, 1), plot_array[0, :])).T
    X, y = fit_array[:, 0], fit_array[:, 1]
    coefficients = np.polyfit(X, y, 5)
    polynomial = np.poly1d(coefficients)

    # MIMIC-IV
    mimic_wdr_res = load_pickle(RESULT_DIR + 'mimic_WDRs_for_ESR.pkl')
    WDR_res = np.array([mimic_wdr_res[i] for i in list(mimic_wdr_res.keys())])
    final_estimated_surv_rate = polynomial(WDR_res)
    save_pickle(final_estimated_surv_rate, RESULT_DIR + 'mimic_ESR.pkl')

    # eICU
    eicu_wdr_res = load_pickle(RESULT_DIR + 'eicu_WDRs_for_ESR.pkl')
    WDR_res = np.array([eicu_wdr_res[i] for i in list(eicu_wdr_res.keys())])
    final_estimated_surv_rate = polynomial(WDR_res)
    save_pickle(final_estimated_surv_rate, RESULT_DIR + 'eicu_ESR.pkl')


def Q_cut_stat_surv_rate(df):
    threshold = np.arange(interval_left, interval_right, 1).tolist()
    df['group'] = pd.cut(df['Q_s_a'], bins=threshold, labels=list(range(len(threshold) - 1)))
    gps = df.groupby('group')
    survival_rate = []
    for group_idx, group in gps:
        survival_rate.append(1 - group['outcome'].sum() / group.shape[0])  # 1是死亡，0是生存
    return survival_rate


if __name__ == '__main__':
    surv_rates = []
    val_Q_with_outcome = load_pickle(RESULT_DIR + '/val-Q_with_outcomes.pkl')
    test_Q_with_outcome = load_pickle(RESULT_DIR + '/test-Q_with_outcomes.pkl')
    for fold_idx in tqdm(range(folds)):
        fold_label = 'fold-{}'.format(fold_idx + 1)
        Q_with_outcome = pd.concat([val_Q_with_outcome[fold_label], test_Q_with_outcome[fold_label]])
        surv_rates.append(Q_cut_stat_surv_rate(Q_with_outcome))
    plot_surv_rates(np.array(surv_rates))
