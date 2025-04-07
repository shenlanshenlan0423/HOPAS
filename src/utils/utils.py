# -*- coding: utf-8 -*-
"""
@Auth ： Hongwei
@File ：utils.py
@IDE ：PyCharm
"""
from definitions import *


class DatasetReconstruction(data.Dataset):
    def __init__(self, tensor_tuple):
        """
        :param tensor_tuple: (states, actions, lengths, LongTermOutcome)
        """
        self.states = tensor_tuple[0]
        self.actions = tensor_tuple[1]
        self.lengths = tensor_tuple[2]
        self.LongTermOutcome = tensor_tuple[3]

    def __getitem__(self, idx):
        done = torch.zeros_like(self.LongTermOutcome[idx])
        nonzero_index = self.LongTermOutcome[idx] != 0
        done[nonzero_index] = 1
        return self.states[idx], self.actions[idx], self.lengths[idx], self.LongTermOutcome[idx], done

    def __len__(self):
        return len(self.states)


def prepare_tuples(tensor_tuple):
    """
    Note: For seq2seq
    tensor_tuple: Tensor: (states, acts, lengths, outcomes)
    :return a full moment of patient data: (states, actions, next_states, lengths, outcomes)
    """
    s_a_ns_len_outcome = []
    Sepsis_data = DatasetReconstruction(tensor_tuple)
    for patient_idx in range(len(Sepsis_data)):
        states, actions, lengths, reward, _ = Sepsis_data[patient_idx]
        next_states = torch.cat((states[1:], torch.zeros((1, states.shape[1])).to(device)), dim=0)
        # Terminal state向自己转移
        next_states[lengths - 1] = states[lengths - 1]
        outcomes = torch.zeros_like(reward).to(device)
        # outcomes[:lengths] = torch.zeros_like(outcomes[:lengths]) if reward[lengths - 1] == 1 else torch.ones_like(outcomes[:lengths])
        outcomes[lengths - 1] = 0 if reward[lengths - 1] == 1 else 1  # 在最后一个时刻，患者产生生存或死亡的结局
        # outcomes = torch.tensor([0]).to(device) if reward[lengths - 1] == 1 else torch.tensor([1]).to(device)
        s_a_ns_len_outcome.append((states, actions, next_states, lengths.unsqueeze(0), outcomes))  # HACK
    return s_a_ns_len_outcome


def Physiological_distance_kernel(X_arr):
    """
    Note: physiological distance kernel for measuring the distance of state.
    Informative features were the patient’s SOFA score, lactate levels, fluid output, mean and diastolic blood pressure,
    PaO2/FiO2 ratio, chloride levels, weight, and age.
    static_cols = ['Age', 'Gender', 'Weight', 'Height']
    temporal_cols = [
        # Vital signs
        'MBP', 'SBP', 'DBP', 'HR', 'Temperature', 'RR', 'SpO2',
        # Lab Test
        'Hb', 'Platelets', 'WBC', 'Hematocrit', 'RDW',  # 'MCH', 'MCHC', 'MCV',
        'PTT', 'INR', 'PH', 'PaO2', 'PaCO2', 'Lactate', 'BaseExcess',  # 'PT',
        'HCO3', 'Chloride', 'Sodium', 'Potassium', 'Glucose',  # 'Calcium', 'FiO2',
        'AnionGap', 'BUN', 'Creatinine',  # 'Albumin',
        # Treatment administration
        'Total Input', 'Total Output', 'Vasopressor', 'Ventilation',
        'SOFA'
    ]
    Feature index: {Platelets: 12, RDW: 15, PTT: 16, INR: 17, Lactate: 21, SOFA: 35}
    """
    interested_col = [12, 15, 16, 17, 21, 35]
    X_arr[:, interested_col] = X_arr[:, interested_col] * np.sqrt(2)
    return X_arr


def star_by_p_value(p_value, String):
    if float(p_value) <= 0.001:
        newString = String + '***'  # + ' ({:.4f})'.format(p_value)
    elif float(p_value) <= 0.05:
        newString = String + '**'
    elif float(p_value) <= 0.01:
        newString = String + '*'
    else:
        newString = String
    return newString


def paired_t_test(proposed_model_values, other_model_values):
    t_statistic, p_value = stats.ttest_ind(proposed_model_values, other_model_values)
    return t_statistic, p_value


def softmax(arr):
    exp_arr = np.exp(arr - np.max(arr))
    softmax_arr = exp_arr / np.sum(exp_arr)
    return softmax_arr
