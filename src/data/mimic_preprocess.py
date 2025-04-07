# -*- coding: utf-8 -*-
"""
@Time ： 2024/12/3 10:27
@Auth ： Hongwei
@File ：mimic_preprocess.py
@IDE ：PyCharm
"""
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from definitions import *

mimic_new_col_names = [
    'traj', 'step',
    # Demographics
    'Age', 'Gender', 'Race', 'Weight', 'Height', 'CCI',
    # Vital signs
    'MBP', 'SBP', 'DBP', 'HR', 'Temperature', 'RR', 'SpO2',
    # Lab Test
    'Hb', 'Platelets', 'WBC', 'Hematocrit', 'MCH', 'MCHC', 'MCV', 'RDW',
    'PT', 'PTT', 'INR', 'PH', 'PaO2', 'PaCO2', 'Lactate', 'BaseExcess',
    'HCO3', 'Chloride', 'Sodium', 'Potassium', 'Calcium', 'Glucose', 'FiO2',
    'Albumin', 'Globulin', 'Total Protein', 'AnionGap', 'BUN', 'Creatinine',
    # Treatment and Scores
    'Total Input', 'Total Output', 'Vasopressor', 'Ventilation',
    'SOFA', 'SIRS', 'APSIII', 'Total Heparin',
    # Length of Stay
    'Los_hospital', 'Los_icu', 'Mort_day',
    # Outcome (Yes/No)
    '28d mortality', '60d mortality', '90d mortality', '1year mortality'
]


def mimic_aggregate_data(traj, gp, granularity=6):
    # 每 granularity hours 小时进行一次聚合
    gp = gp.reset_index().assign(hour=lambda x: x.index // granularity)
    sub_gps = gp.groupby('hour')
    mean_aggregate_cols = [
        'Age', 'Gender', 'Race', 'Weight', 'Height', 'CCI',
        'MBP', 'SBP', 'DBP', 'HR', 'Temperature', 'RR', 'SpO2',
        'Hb', 'Platelets', 'WBC', 'Hematocrit', 'MCH', 'MCHC', 'MCV', 'RDW',
        'PT', 'PTT', 'INR', 'PH', 'PaO2', 'PaCO2', 'Lactate', 'BaseExcess',
        'HCO3', 'Chloride', 'Sodium', 'Potassium', 'Calcium', 'Glucose', 'FiO2',
        'Albumin', 'Globulin', 'Total Protein', 'AnionGap', 'BUN', 'Creatinine',
        'SOFA', 'SIRS', 'APSIII',
        'Los_hospital', 'Los_icu', 'Mort_day',
        '28d mortality', '60d mortality', '90d mortality', '1year mortality'
    ]
    sum_aggregate_cols = [
        'Total Input', 'Total Output', 'Total Heparin',
    ]
    max_aggregate_cols = [
        'Vasopressor', 'Ventilation'
    ]
    aggregated_df = pd.DataFrame(columns=mimic_new_col_names)
    aggregated_df['traj'] = [traj] * len(sub_gps)
    aggregated_df['step'] = sub_gps['hour'].mean().values
    aggregated_df[mean_aggregate_cols] = sub_gps[mean_aggregate_cols].mean().values
    aggregated_df[sum_aggregate_cols] = sub_gps[sum_aggregate_cols].sum().values
    aggregated_df[max_aggregate_cols] = sub_gps[max_aggregate_cols].max().values
    return aggregated_df


def mimic_fill_patient_nan(gp):
    """
    插补完后还存在的nan是由于该患者的此列完全缺失，因此不会涉及到患者step的打乱
    """
    mean_fill_cols = [
        'Weight', 'Height'
    ]
    forward_fill_cols = [
        'MBP', 'SBP', 'DBP', 'HR', 'Temperature', 'RR', 'SpO2',
        'Hb', 'Platelets', 'WBC', 'Hematocrit', 'MCH', 'MCHC', 'MCV', 'RDW',
        'PT', 'PTT', 'INR', 'PH', 'PaO2', 'PaCO2', 'Lactate', 'BaseExcess',
        'HCO3', 'Chloride', 'Sodium', 'Potassium', 'Calcium', 'Glucose', 'FiO2',
        'Albumin', 'Globulin', 'Total Protein', 'AnionGap', 'BUN', 'Creatinine',
    ]
    gp[mean_fill_cols] = gp[mean_fill_cols].fillna(gp[mean_fill_cols].mean(0))
    gp[forward_fill_cols] = gp[forward_fill_cols].fillna(method='ffill')
    # gp[forward_fill_cols] = gp[forward_fill_cols].fillna(gp[forward_fill_cols].mean(0))
    # 先forward fill插补，再backward fill
    gp[forward_fill_cols] = gp[forward_fill_cols].fillna(method='bfill')
    return gp


def mimic_fill_demographics(mimic_filled):
    """
    基于'Age', 'Gender', 'Race', 'CCI'做match (1:300)，再对'Weight', 'Height'进行插补
    """
    no_Demographics_nan_df = mimic_filled[['Age', 'Gender', 'Race', 'Weight', 'Height', 'CCI']].dropna(
        subset=['Weight', 'Height'])
    no_Demographics_nan_df_index = no_Demographics_nan_df.index
    no_Demographics_nan_df = no_Demographics_nan_df.reset_index(drop=True)
    encoded_no_Demographics_nan_df = pd.get_dummies(no_Demographics_nan_df, columns=['Gender', 'Race'])
    scaled_no_Demographics_nan_df = np.hstack(
        [stats.zscore(encoded_no_Demographics_nan_df[['Age', 'CCI']].values),
         encoded_no_Demographics_nan_df.values[:, 4:]]
    )
    neigh = NearestNeighbors(n_neighbors=300)
    neigh.fit(scaled_no_Demographics_nan_df)

    # Weight和Height起码有一个为空的行
    need_fill_row = mimic_filled[mimic_filled[['Weight', 'Height']].isna().any(axis=1)][
        ['Age', 'Gender', 'Race', 'Weight', 'Height', 'CCI']]
    need_fill_row_index = need_fill_row.index
    need_fill_row = need_fill_row.reset_index(drop=True)

    encoded_need_fill_row = pd.get_dummies(need_fill_row, columns=['Gender', 'Race'])
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
    mimic_filled[['Weight', 'Height']] = final_values.values
    return mimic_filled


if __name__ == '__main__':
    mimic_table = pd.read_csv(DATA_DIR + '/SIT_temporal_exclusion_1127.csv')
    new_heparin_table = pd.read_csv(DATA_DIR + '/temporal_heparin_detail_1205.csv')

    print('\n----------------------------------------异常值处理----------------------------------------\n')
    # 液体总入量=胶体+晶体
    mimic_table = mimic_table.merge(new_heparin_table, on=['stay_id', 'hr'], how='left')  # 13355名患者
    mimic_table['TotalFluidInput'] = mimic_table['total_crystalloid_bolus'] + mimic_table['total_colloid_bolus']
    filtered_cols = [
        'stay_id', 'hr',
        'age', 'gender', 'race', 'weight', 'height', 'charlson_comorbidity_index',
        'mbp', 'sbp', 'dbp', 'heart_rate', 'temperature', 'resp_rate', 'spo2',
        'hemoglobin', 'platelet', 'wbc', 'hematocrit', 'mch', 'mchc', 'mcv', 'rdw',
        'pt', 'ptt', 'inr', 'ph', 'po2', 'pco2', 'lactate', 'baseexcess',
        'bicarbonate', 'chloride', 'sodium', 'potassium', 'calcium', 'glucose', 'fio2',
        'albumin', 'globulin', 'total_protein', 'aniongap', 'bun', 'creatinine',
        'TotalFluidInput', 'total_urine_output', 'total_norepinephrine_dose', 'invasive_vent',
        'sofa_temporal', 'sirs_score', 'apsiii', 'heparin_total_y',
        'los_hospital', 'los_icu', 'mort_day',
        'day28_expired_flag', 'day60_expired_flag', 'day90_expired_flag', 'day365_expired_flag'
    ]
    mimic_filtered = mimic_table[filtered_cols]
    mimic_filtered.columns = mimic_new_col_names
    # 部分体重记录错误
    mimic_filtered.loc[mimic_filtered['Weight'] == 1, 'Weight'] *= 100
    # Total Output为负的部分可能是由记录错误引起
    mimic_filtered.loc[(mimic_filtered['Total Output'] < 0), 'Total Output'] = -mimic_filtered.loc[(mimic_filtered['Total Output'] < 0), 'Total Output']
    # 对于超出具备临床意义范围的特征取值，替换为nan
    for feature, interval in [('MBP', [20, 160]), ('SBP', [40, 250]), ('DBP', [20, 160]),
                              ('HR', [20, 250]), ('Temperature', [23.7, 44]), ('RR', [3, 80]), ('SpO2', [40, 100]),
                              ('Hb', [1, 30]), ('Platelets', [1, 1500]), ('WBC', [0.01, 300]), ('Hematocrit', [5, 75]),
                              ('MCH', [10, 50]), ('MCHC', [15, 50]), ('MCV', [40, 150]), ('RDW', [5, 50]),
                              ('PT', [5, 200]), ('PTT', [10, 200]), ('INR', [0.3, 21.5]), ('PH', [6.5, 8.0]),  # ('INR', [0.3, 20])
                              ('PaO2', [10, 1000]), ('PaCO2', [5, 200]), ('Lactate', [0.1, 50]),
                              ('BaseExcess', [-35, 42]), ('HCO3', [2, 70]), ('Chloride', [63, 155]),
                              ('Sodium', [80, 200]), ('Potassium', [1, 20]), ('Calcium', [0.5, 40]),
                              ('Glucose', [10, 2010]), ('FiO2', [21, 100]),
                              ('Albumin', [0.5, 6]), ('Globulin', [0.2, 8]), ('Total Protein', [1, 12]),
                              ('AnionGap', [-10, 60]), ('BUN', [1, 300]), ('Creatinine', [0.1, 50]),
                              ('Total Input', [0, 10000]), ('Total Output', [0, 15000]),
                              ('Vasopressor', [0, 100]), ('Total Heparin', [0, 50000])]:
        mimic_filtered.loc[(mimic_filtered[feature] < interval[0]) | (mimic_filtered[feature] > interval[1]), feature] = pd.np.nan

    print('\n----------------------------------------数据聚合----------------------------------------\n')
    mimic_filtered = mimic_filtered[mimic_filtered['step'] != 72].reset_index(drop=True)  # 第72小时的存在导致总时刻为24+1+72=97
    gps = mimic_filtered.groupby('traj')
    new_data_list = []
    for traj, gp in tqdm(gps):
        aggregated_gp = mimic_aggregate_data(traj, gp)
        new_data_list.extend(aggregated_gp.values.tolist())
    mimic_aggregated = pd.DataFrame(np.array(new_data_list), columns=mimic_new_col_names)
    mimic_aggregated.to_csv(DATA_DIR + 'mimic_aggregated.csv', index=False)

    print('\n----------------------------------------数据插补----------------------------------------\n')
    mimic_aggregated = pd.read_csv(DATA_DIR + 'mimic_aggregated.csv')
    # print(mimic_aggregated.info())
    gps = mimic_aggregated.groupby('traj')
    new_data_list = []
    for traj, gp in tqdm(gps):
        filled_gp = mimic_fill_patient_nan(gp)
        new_data_list.extend(filled_gp.values.tolist())
    mimic_patient_filled = pd.DataFrame(np.array(new_data_list), columns=mimic_new_col_names)
    # 删除仍有较多缺失的特征
    mimic_patient_filled.drop(labels=['Globulin', 'Total Protein'], axis=1, inplace=True)
    # 对于Weight和Height缺失的state，利用KNN找300个最近邻的Weight或Height的均值进行插值 (基于其他Demographics计算距离)
    mimic_demo_filled = mimic_fill_demographics(mimic_patient_filled)
    # 90d 结局相同的患者，进行均值插补
    gps = mimic_demo_filled.groupby('traj')
    outcome_90d = gps['90d mortality'].mean()
    dead_patients = pd.concat([gps.get_group(id_) for id_ in outcome_90d[outcome_90d == 1].index.tolist()]).reset_index()
    dead_patients_full_filled = dead_patients.fillna(dead_patients.mean(0))
    survival_patients = pd.concat([gps.get_group(id_) for id_ in outcome_90d[outcome_90d == 0].index.tolist()]).reset_index()
    survival_patients_full_filled = survival_patients.fillna(survival_patients.mean(0))
    mimic_full_filled = pd.concat([survival_patients_full_filled, dead_patients_full_filled], axis=0).set_index('index').sort_index()

    print('\n----------------------------------------MDP settings----------------------------------------\n')
    # 使用映射字典替换 'traj' 列的值
    traj_to_code = {traj: code for code, traj in enumerate(mimic_full_filled['traj'].unique().tolist(), start=1)}
    mimic_full_filled['traj'] = mimic_full_filled['traj'].map(traj_to_code)
    mimic_full_filled['Mort_day'] = mimic_aggregated['Mort_day']  # Note: 这里的结局标签不用插补
    mimic_full_filled.to_csv(DATA_DIR + 'mimic-iv-sepsis_RAW_withTimesTrueDosage.csv', index=False)
    # Note: ----------------Action----------------
    mimic_full_filled = pd.read_csv(DATA_DIR + 'mimic-iv-sepsis_RAW_withTimesTrueDosage.csv')
    raw_heparin = mimic_full_filled['Total Heparin'].values
    summary_stats = pd.DataFrame(
        raw_heparin[(mimic_full_filled['Total Heparin'] > 0) & (mimic_full_filled['Total Heparin'] != 1500)]).describe(
        percentiles=[0.25, 0.5, 0.75])
    # MIMIC-IV: [-1, 0, 3600.0, 5402.161010742188, 7464.140504964193, 42645.373787434895]
    mimic_full_filled['action'] = pd.cut(mimic_full_filled['Total Heparin'],
                                         bins=[-1, 0, summary_stats.loc['25%'].values[0],
                                               summary_stats.loc['50%'].values[0],
                                               summary_stats.loc['75%'].values[0],
                                               summary_stats.loc['max'].values[0]],
                                         labels=[0, 1, 2, 3, 4], right=True)  # right=True(默认值)时,区间是左开右闭
    # Note: ----------------Reward----------------
    gps = mimic_full_filled.groupby('traj')
    rewards_90d = []
    for patient, gp in gps:
        reward = [0] * gp.shape[0]
        reward[-1] = 1 - 2 * gp['90d mortality'].mean()
        rewards_90d.extend(reward)
    mimic_full_filled['rewards_90d'] = rewards_90d
    # mimic_full_filled.drop(labels=['Total Heparin'], axis=1, inplace=True)
    mimic_full_filled.to_csv(DATA_DIR + 'mimic-iv-sepsis_RAW_withTimes.csv', index=False)

    print('\n----------------------------------------数据标准化----------------------------------------\n')
    mimic_raw = pd.read_csv(DATA_DIR + 'mimic-iv-sepsis_RAW_withTimes.csv')
    colmeta = ['traj', 'step']  # 2
    # Binary features  2
    colbin = ['Gender', 'Ventilation']
    # Features that will be z-normalize (适用于数据呈现正态分布或接近正态分布；需要消除特征之间的量纲差异。) 32
    colnorm = ['Age', 'Weight', 'Height', 'CCI',  # 4
               'MBP', 'SBP', 'DBP', 'HR', 'Temperature', 'RR', 'SpO2',  # 7
               'Hb', 'Platelets', 'WBC', 'Hematocrit', 'MCH', 'MCHC', 'MCV',  # 7
               'PH', 'PaCO2', 'BaseExcess',  # 3
               'HCO3', 'Chloride', 'Sodium', 'Potassium', 'Calcium', 'Glucose',  'FiO2',  # 7
               'Albumin', 'AnionGap', 'SOFA', 'APSIII'  # 4
               ]
    # Features that will be log-normalize (适用于数据呈现正偏分布,即右侧尾部长；数据包含较大差异的正值,需要缩小高值与低值间的差距。) 12
    collog = ['RDW', 'PT', 'PTT', 'INR', 'PaO2', 'Lactate', 'BUN', 'Creatinine', 'Total Input', 'Total Output', 'Vasopressor', 'SIRS']
    mimic_zs = pd.DataFrame(np.hstack(
        [mimic_raw[colmeta].values, mimic_raw[colbin].values - 0.5,
         stats.zscore(mimic_raw[colnorm].values),
         stats.zscore(np.log(0.0001 + mimic_raw[collog].values)),   # 保证log的真数 > 0
         mimic_raw[['rewards_90d', 'action']].values]
    ), columns=colmeta + colbin + colnorm + collog + ['rewards_90d', 'action'])
    mimic_zs[['Los_hospital', 'Los_icu', '28d mortality', '60d mortality', '90d mortality', '1year mortality']] = \
        mimic_raw[['Los_hospital', 'Los_icu', '28d mortality', '60d mortality', '90d mortality', '1year mortality']]
    mimic_zs.to_csv(DATA_DIR + 'mimic-iv-sepsis_withTimes.csv', index=False)
