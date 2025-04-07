# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/30 11:06
@Auth ： Hongwei
@File ：dataset_des.py
@IDE ：PyCharm
"""
from definitions import *

if __name__ == '__main__':
    mimic_table = pd.read_csv(DATA_DIR + '/mimic-iv-sepsis_RAW_withTimesTrueDosage.csv')
    eicu_table = pd.read_csv(DATA_DIR + '/eicu-sepsis_RAW_withTimesTrueDosage.csv')

    df_two_dataset_des = []
    for df, outcome_label in zip([mimic_table, eicu_table], ['90d mortality', 'ICU mortality']):
        gps = df.groupby('traj')
        Age, Gender, Weight, Height, Outcome = [], [], [], [], []
        for idx, gp in tqdm(gps):
            Age.append(gp['Age'].mean())
            Gender.append(gp['Gender'].mean())
            Weight.append(gp['Weight'].mean())
            Height.append(gp['Height'].mean())
            Outcome.append(gp[outcome_label].mean())

        patient_numer = len(gps)
        demos = np.array([['Unique ICU admissions (n)', 'All states', 'Age', 'Male (n(%))', 'Weight', 'Height'],
                         [patient_numer, df.shape[0],
                          '{:.2f}±{:.2f}'.format(np.mean(Age), np.std(Age)),
                          '{} ({:.2f}%)'.format(np.count_nonzero(np.array(Gender) == 0), np.count_nonzero(np.array(Gender) == 0) / patient_numer * 100),
                          '{:.2f}±{:.2f}'.format(np.mean(Weight), np.std(Weight)),
                          '{:.2f}±{:.2f}'.format(np.mean(Height), np.std(Height))]]).T

        temporal_col_list = []
        for col in temporal_cols:
            temporal_col_list.append('{:.2f}±{:.2f}'.format(np.mean(df[col]), np.std(df[col])))
        temporal_variables = np.array([temporal_cols, temporal_col_list]).T

        mechvent = np.array(
            [['Mechanical ventilation (n(%))'],
             ['{} ({:.2f}%)'.format(np.count_nonzero(np.array(df['Ventilation']) == 1), np.count_nonzero(np.array(df['Ventilation']) == 1) / df.shape[0] * 100)]]
        ).T

        outcome = np.array(
            [['{} (n(%))'.format(outcome_label)],
             ['{} ({:.2f}%)'.format(np.count_nonzero(np.array(Outcome) == 1), np.count_nonzero(np.array(Outcome) == 1) / patient_numer * 100)]]).T
        feature_des = pd.DataFrame(np.vstack((demos, temporal_variables, outcome)))
        feature_des.loc[36] = mechvent  # replace the wrong description of ventilation
        df_two_dataset_des.append(feature_des.values.tolist())
    final_des_table = pd.DataFrame(np.array(df_two_dataset_des)[0], columns=['Features', 'MIMIC-IV'])
    final_des_table['eICU'] = np.array(df_two_dataset_des)[1][:, 1]
    new_col_names = ['Unique ICU admissions (n)', 'All states (n)', 'Age', 'Male (n(%))', 'Weight', 'Height',
                     'MeanBP', 'SysBP', 'DiasBP', 'Heart rate', 'Temperature', 'Respratory', 'Spo2',
                     'Hemoglobin', 'Platelet', 'WBC', 'Hematocrit', 'RDW', 'aPTT', 'INR', 'PH', 'PaO2', 'PaCO2',
                     'Lactate', 'Base excess', 'Bicarbonate', 'Chloride', 'Sodium', 'Potassium', 'Glucose', 'Anion gap',
                     'BUN', 'Creatinine', 'Total input', 'Urine output', 'Vasopressor', 'Mechanical ventilation (n(%))',
                     'SOFA score',
                     '90d mortality (n(%))', # 'ICU mortality (n(%))'
                     ]
    final_des_table['Features'] = new_col_names
    final_des_table.to_excel(TABLE_DIR + 'feature_des.xlsx', index=False)
