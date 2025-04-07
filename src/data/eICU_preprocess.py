# -*- coding: utf-8 -*-
"""
@Time ： 2024/12/3 10:27
@Auth ： Hongwei
@File ：eicu_preprocess.py
@IDE ：PyCharm
"""
from definitions import *
from src.data.mimic_preprocess import NearestNeighbors, stats

eicu_new_col_names = [
    'traj', 'step',
    # Demographics
    'Age', 'Gender', 'Weight', 'Height', 'CCI',
    # Vital signs
    'MBP', 'SBP', 'DBP', 'HR', 'Temperature', 'RR', 'SpO2',
    # Lab Test
    'Hb', 'Platelets', 'WBC', 'Hematocrit', 'RDW',
    'PTT', 'INR', 'PH', 'PaO2', 'PaCO2', 'Lactate', 'BaseExcess',
    'HCO3', 'Chloride', 'Sodium', 'Potassium', 'Glucose',
    'AnionGap', 'BUN', 'Creatinine',
    # Treatment and Scores
    'Total Input', 'Total Output', 'Vasopressor', 'Ventilation',
    'SOFA', 'Total Heparin',
    # Outcome (Yes/No)
    'ICU mortality', 'Hosp mortality'
]


def eicu_aggregate_data(traj, gp, granularity=6):
    # 每 granularity hours 小时进行一次聚合
    gp = gp.reset_index().assign(hour=lambda x: x.index // granularity)
    sub_gps = gp.groupby('hour')
    mean_aggregate_cols = [
        'Age', 'Gender', 'Weight', 'Height', 'CCI',
        'MBP', 'SBP', 'DBP', 'HR', 'Temperature', 'RR', 'SpO2',
        'Hb', 'Platelets', 'WBC', 'Hematocrit', 'RDW',
        'PTT', 'INR', 'PH', 'PaO2', 'PaCO2', 'Lactate', 'BaseExcess',
        'HCO3', 'Chloride', 'Sodium', 'Potassium', 'Glucose',
        'AnionGap', 'BUN', 'Creatinine', 'SOFA',
        'ICU mortality', 'Hosp mortality'
    ]
    sum_aggregate_cols = [
        'Total Input', 'Total Output', 'Total Heparin',
    ]
    max_aggregate_cols = [
        'Vasopressor', 'Ventilation'
    ]
    aggregated_df = pd.DataFrame(columns=eicu_new_col_names)
    aggregated_df['traj'] = [traj] * len(sub_gps)
    aggregated_df['step'] = sub_gps['hour'].mean().values
    aggregated_df[mean_aggregate_cols] = sub_gps[mean_aggregate_cols].mean().values
    aggregated_df[sum_aggregate_cols] = sub_gps[sum_aggregate_cols].sum().values
    aggregated_df[max_aggregate_cols] = sub_gps[max_aggregate_cols].max().values
    return aggregated_df


def eicu_fill_patient_nan(gp):
    """
    插补完后还存在的nan是由于该患者的此列完全缺失，因此不会涉及到患者step的打乱
    """
    mean_fill_cols = [
        'Weight', 'Height'
    ]
    forward_fill_cols = [
        'MBP', 'SBP', 'DBP', 'HR', 'Temperature', 'RR', 'SpO2',
        'Hb', 'Platelets', 'WBC', 'Hematocrit', 'RDW',
        'PTT', 'INR', 'PH', 'PaO2', 'PaCO2', 'Lactate', 'BaseExcess',
        'HCO3', 'Chloride', 'Sodium', 'Potassium', 'Glucose',
        'AnionGap', 'BUN', 'Creatinine', 'SOFA'
    ]
    gp[mean_fill_cols] = gp[mean_fill_cols].fillna(gp[mean_fill_cols].mean(0))
    gp[forward_fill_cols] = gp[forward_fill_cols].fillna(method='ffill')
    # gp[forward_fill_cols] = gp[forward_fill_cols].fillna(gp[forward_fill_cols].mean(0))
    # 先forward fill插补，再backward fill
    gp[forward_fill_cols] = gp[forward_fill_cols].fillna(method='bfill')
    return gp


def eicu_fill_demographics(eicu_filled):
    """
    基于'Age', 'Gender', 'CCI'做 match (1:300)，再对'Weight', 'Height'进行插补
    """
    no_Demographics_nan_df = eicu_filled[['Age', 'Gender', 'Weight', 'Height', 'CCI']].dropna(
        subset=['Weight', 'Height'])
    no_Demographics_nan_df_index = no_Demographics_nan_df.index
    no_Demographics_nan_df = no_Demographics_nan_df.reset_index(drop=True)
    encoded_no_Demographics_nan_df = pd.get_dummies(no_Demographics_nan_df, columns=['Gender'])
    scaled_no_Demographics_nan_df = np.hstack(
        [stats.zscore(encoded_no_Demographics_nan_df[['Age', 'CCI']].values),
         encoded_no_Demographics_nan_df.values[:, 4:]]
    )
    neigh = NearestNeighbors(n_neighbors=300)
    neigh.fit(scaled_no_Demographics_nan_df)

    # Weight和Height起码有一个为空的行
    need_fill_row = eicu_filled[eicu_filled[['Weight', 'Height']].isna().any(axis=1)][
        ['Age', 'Gender', 'Weight', 'Height', 'CCI']]
    need_fill_row_index = need_fill_row.index
    need_fill_row = need_fill_row.reset_index(drop=True)

    encoded_need_fill_row = pd.get_dummies(need_fill_row, columns=['Gender'])
    scaled_need_fill_row = np.hstack(
        [stats.zscore(encoded_need_fill_row[['Age', 'CCI']].values),
         encoded_need_fill_row.values[:, 4:]]
    )
    distances, indices = neigh.kneighbors(scaled_need_fill_row)
    neighbors_Weights_Height = no_Demographics_nan_df.values[indices][:, :, [3, 4]]
    need_fill_row[['neighbors mean Weight', 'neighbors mean Height']] = neighbors_Weights_Height.mean(axis=1)
    # 有nan进行填充，没nan用原来的
    need_fill_row['Final Weight'] = need_fill_row['Weight'].where(need_fill_row['Weight'].notnull(),
                                                                  need_fill_row['neighbors mean Weight'])
    need_fill_row['Final Height'] = need_fill_row['Height'].where(need_fill_row['Height'].notnull(),
                                                                  need_fill_row['neighbors mean Height'])

    final_values = pd.DataFrame(np.vstack((
        np.hstack((no_Demographics_nan_df[['Weight', 'Height']].values,
                   np.array(no_Demographics_nan_df_index).reshape(-1, 1))),
        np.hstack(
            (need_fill_row[['Final Weight', 'Final Height']].values, np.array(need_fill_row_index).reshape(-1, 1)))
    )), columns=['Weight', 'Height', 'index']).set_index('index').sort_index()
    eicu_filled[['Weight', 'Height']] = final_values.values
    return eicu_filled


if __name__ == '__main__':
    eicu_table = pd.read_csv(DATA_DIR + '/eicu_continuousdata_1hr_withlabelnew_withsc_modvent.csv')
    pass
    print('\n----------------------------------------异常值处理----------------------------------------\n')
    filtered_cols = [
        'traj', 'relative_hour',
        'Age', 'Gender', 'Weight', 'Height', 'CCI',
        'MeanBP', 'SystolicBP', 'DiastolicBP', 'HR', 'Temperature', 'RR', 'SpO2',
        'Hb', 'Platelets', 'WBC', 'Hematocrit', 'RDW',
        'PTT', 'INR', 'PH', 'PaO2', 'PaCO2', 'ArterialLactate', 'ArterialBE',
        'HCO3', 'Chloride', 'Sodium', 'Potassium', 'Glucose',
        'AnionGap', 'BUN', 'Creatinine',

        'TotalFluidInput', 'Total output', 'TotalNorepinephrineDose', 'Invasive_vent',
        'SOFA', 'Total_heparin',
        # Outcome (Yes/No)
        'ICU_mortality', 'Hosp_mortality'
    ]
    eicu_filtered = eicu_table[filtered_cols]  # 6066名患者
    eicu_filtered.columns = eicu_new_col_names
    eicu_filtered = eicu_filtered.drop_duplicates()  # 去除重复行
    pass
    # Total Input和Total Output为负的部分可能是由记录错误引起
    eicu_filtered.loc[eicu_filtered['Total Input'] < 0, 'Total Input'] *= -1
    eicu_filtered.loc[eicu_filtered['Total Output'] < 0, 'Total Output'] *= -1
    # 部分医院身高记录单位不同，统一为cm
    eicu_filtered.loc[eicu_filtered['Height'] < 10, 'Height'] *= 100
    eicu_filtered.loc[(eicu_filtered['Height'] >= 10) & (eicu_filtered['Height'] < 20), 'Height'] *= 10
    # 对于超出具备临床意义范围的特征取值，替换为nan
    for feature, interval in [('Weight', [18, 500]), ('MBP', [20, 160]), ('SBP', [40, 250]), ('DBP', [20, 160]),
                              ('HR', [20, 250]), ('Temperature', [23.7, 44]), ('RR', [3, 80]), ('SpO2', [40, 100]),
                              ('Hb', [1, 30]), ('Platelets', [1, 1500]), ('WBC', [0.01, 300]), ('Hematocrit', [5, 75]),
                              ('RDW', [5, 50]), ('PTT', [10, 200]), ('INR', [0.3, 21.5]), ('PH', [6.5, 8.0]),
                              ('PaO2', [10, 1000]), ('PaCO2', [5, 200]), ('Lactate', [0.1, 50]),
                              ('BaseExcess', [-35, 42]), ('HCO3', [2, 70]), ('Chloride', [63, 155]),
                              ('Sodium', [80, 200]), ('Potassium', [1, 20]), ('Glucose', [10, 2010]),
                              ('AnionGap', [-10, 60]), ('BUN', [1, 300]), ('Creatinine', [0.1, 50]),
                              ('Total Input', [0, 10000]), ('Total Output', [0, 15000]),
                              ('Vasopressor', [0, 100]), ('Total Heparin', [0, 50000])]:
        eicu_filtered.loc[(eicu_filtered[feature] < interval[0]) | (eicu_filtered[feature] > interval[1]), feature] = np.nan
        # pd.np.nan
    print('\n----------------------------------------数据聚合----------------------------------------\n')
    eicu_filtered = eicu_filtered[eicu_filtered['step'] != 72].reset_index(drop=True)  # 第72小时的存在导致总时刻为24+1+72=97
    gps = eicu_filtered.groupby('traj')
    new_data_list = []
    for traj, gp in tqdm(gps):
        aggregated_gp = eicu_aggregate_data(traj, gp)
        new_data_list.extend(aggregated_gp.values.tolist())
    eicu_aggregated = pd.DataFrame(np.array(new_data_list), columns=eicu_new_col_names)
    eicu_aggregated.to_csv(DATA_DIR + 'eicu_aggregated.csv', index=False)

    print('\n----------------------------------------数据插补----------------------------------------\n')
    eicu_aggregated = pd.read_csv(DATA_DIR + 'eicu_aggregated.csv')
    eicu_aggregated['Gender'] = 1 - eicu_aggregated['Gender']  # 与MIMIC-IV中的性别编码一致
    # print(eicu_aggregated.info())
    gps = eicu_aggregated.groupby('traj')
    new_data_list = []
    for traj, gp in tqdm(gps):
        filled_gp = eicu_fill_patient_nan(gp)
        new_data_list.extend(filled_gp.values.tolist())
    eicu_patient_filled = pd.DataFrame(np.array(new_data_list), columns=eicu_new_col_names)
    # 用药使用零值插补; Note: 死亡结局会特别关注，所以缺失的一般都是存活的
    eicu_patient_filled[['Vasopressor', 'Hosp mortality']] = eicu_patient_filled[['Vasopressor', 'Hosp mortality']].fillna(0)
    # 对于Weight和Height缺失的state，利用KNN找300个最近邻的Weight或Height的均值进行插值 (基于其他Demographics计算距离)
    eicu_demo_filled = eicu_fill_demographics(eicu_patient_filled)
    # 使用映射字典替换 'traj' 列的值
    traj_to_code = {traj: code for code, traj in enumerate(eicu_demo_filled['traj'].unique().tolist(), start=1)}
    eicu_demo_filled['traj'] = eicu_demo_filled['traj'].map(traj_to_code)
    eicu_missing_filtered = eicu_demo_filled
    # icu结局相同的患者，进行均值插补
    gps = eicu_missing_filtered.groupby('traj')
    outcome_90d = gps['ICU mortality'].mean()
    dead_patients = pd.concat([gps.get_group(id_) for id_ in outcome_90d[outcome_90d == 1].index.tolist()]).reset_index()
    dead_patients_full_filled = dead_patients.fillna(dead_patients.mean(0))
    survival_patients = pd.concat([gps.get_group(id_) for id_ in outcome_90d[outcome_90d == 0].index.tolist()]).reset_index()
    survival_patients_full_filled = survival_patients.fillna(survival_patients.mean(0))
    eicu_full_filled = pd.concat([survival_patients_full_filled, dead_patients_full_filled], axis=0).set_index('index').sort_index()
    print('\n----------------------------------------MDP settings----------------------------------------\n')
    # 使用映射字典替换 'traj' 列的值
    traj_to_code = {traj: code for code, traj in enumerate(eicu_full_filled['traj'].unique().tolist(), start=1)}
    eicu_full_filled['traj'] = eicu_full_filled['traj'].map(traj_to_code)
    # eicu_full_filled.to_csv(DATA_DIR + 'eicu-sepsis_RAW_withTimesTrueDosageForDes.csv', index=False)
    eicu_full_filled.to_csv(DATA_DIR + 'eicu-sepsis_RAW_withTimesTrueDosage.csv', index=False)
    # Note: ----------------Action----------------
    eicu_full_filled = pd.read_csv(DATA_DIR + 'eicu-sepsis_RAW_withTimesTrueDosage.csv')
    raw_heparin = eicu_full_filled['Total Heparin'].values
    summary_stats = pd.DataFrame(
        raw_heparin[(eicu_full_filled['Total Heparin'] > 0) & (eicu_full_filled['Total Heparin'] != 1500)]).describe(
        percentiles=[0.25, 0.5, 0.75])
    # eicu: [-1, 0, 4493.333333333333, 6273.5, 8700.0, 229602.0]
    eicu_full_filled['action'] = pd.cut(eicu_full_filled['Total Heparin'],
                                        bins=[-1, 0, summary_stats.loc['25%'].values[0],
                                              summary_stats.loc['50%'].values[0],
                                              summary_stats.loc['75%'].values[0],
                                              summary_stats.loc['max'].values[0]],
                                        labels=[0, 1, 2, 3, 4], right=True)  # right=True(默认值)时,区间是左开右闭
    # Note: ----------------Reward----------------
    gps = eicu_full_filled.groupby('traj')
    rewards_icu, rewards_hosp = [], []
    for patient, gp in gps:
        reward = [0] * gp.shape[0]
        reward[-1] = 1 - 2 * gp['ICU mortality'].mean()
        rewards_icu.extend(reward)
        reward[-1] = 1 - 2 * gp['Hosp mortality'].mean()
        rewards_hosp.extend(reward)
    eicu_full_filled['rewards_icu'] = rewards_icu
    eicu_full_filled['rewards_hosp'] = rewards_hosp
    eicu_full_filled.drop(labels=['Total Heparin', 'ICU mortality', 'Hosp mortality'], axis=1, inplace=True)
    eicu_full_filled.to_csv(DATA_DIR + 'eicu-sepsis_RAW_withTimes.csv', index=False)

    print('\n----------------------------------------数据标准化----------------------------------------\n')
    eicu_raw = pd.read_csv(DATA_DIR + 'eicu-sepsis_RAW_withTimes.csv')

    colmeta = ['traj', 'step']  # 2
    # Binary features  2
    colbin = ['Gender', 'Ventilation']
    # Features that will be z-normalize (适用于数据呈现正态分布或接近正态分布；需要消除特征之间的量纲差异。) 26
    colnorm = ['Age', 'Weight', 'Height', 'CCI',  # 4
               'MBP', 'SBP', 'DBP', 'HR', 'Temperature', 'RR', 'SpO2',  # 7
               'Hb', 'Platelets', 'WBC', 'Hematocrit',  # 4
               'PH', 'PaCO2', 'BaseExcess',  # 3
               'HCO3', 'Chloride', 'Sodium', 'Potassium', 'Glucose',  # 5
               'AnionGap', 'SOFA',  # 2
               ]
    # Features that will be log-normalize (适用于数据呈现正偏分布,即右侧尾部长；数据包含较大差异的正值,需要缩小高值与低值间的差距。) 9
    collog = ['RDW', 'PTT', 'INR', 'PaO2', 'Lactate', 'BUN', 'Creatinine', 'Total Input', 'Total Output', 'Vasopressor']
    eicu_zs = pd.DataFrame(np.hstack(
        [eicu_raw[colmeta].values, eicu_raw[colbin].values - 0.5,
         stats.zscore(eicu_raw[colnorm].values),
         stats.zscore(np.log(0.0001 + eicu_raw[collog].values)),  # 保证log的真数 > 0
         eicu_raw[['rewards_icu', 'rewards_hosp', 'action']].values]
    ), columns=colmeta + colbin + colnorm + collog + ['rewards_icu', 'rewards_hosp', 'action'])
    eicu_zs.to_csv(DATA_DIR + 'eicu-sepsis_withTimes.csv', index=False)
