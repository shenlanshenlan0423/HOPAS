"""
This file contains basic variables and definitions that we wish to make easily accessible for any script that requires
it.

from definitions import *
"""
# Packages/functions used everywhere
from pathlib import Path
import os
import time
from tqdm import tqdm
import copy
import numpy as np
import pandas as pd
import collections
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
from src.omni.functions import load_pickle, save_pickle
from matplotlib import rc
import scipy.stats as stats
import warnings
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import math

rc('font', family='Palatino Linotype')
warnings.filterwarnings('ignore')

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

state_cols = static_cols + temporal_cols

action_col = ['action']
ShortTermOutcome = ['SOFA']
# LongTermOutcome = 90d mortality for MIMIC-IV or ICU mortality for eICU

# Define the features of the full data
state_dim = len(state_cols)
horizon = 16
action_dim = 5

# Type of Evaluation policy
eval_policy_type = 'Stochastic'

folds = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = str(Path(__file__).resolve().parents[0])
DATA_DIR = ROOT_DIR + '/data/'
MODELS_DIR = ROOT_DIR + '/models/'
RESULT_DIR = ROOT_DIR + '/results/'
SOURCE_DIR = ROOT_DIR + '/src/'
FIG_DIR = ROOT_DIR + '/FigSupp/'
TABLE_DIR = ROOT_DIR + '/TableSupp/'

if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(SOURCE_DIR, exist_ok=True)
    os.makedirs(SOURCE_DIR + '/data/', exist_ok=True)
    os.makedirs(SOURCE_DIR + '/model/', exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
