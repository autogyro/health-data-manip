

import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#import local modules
from importlib.machinery import SourceFileLoader
mod_path = "/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/nhanes2013-2104/"
txt = SourceFileLoader("text_utils", mod_path+"text_utils.py").load_module()
gut = SourceFileLoader("graph_utils", mod_path+"graph_utils.py").load_module()


#Path to datasets folders

datasets_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/'
project_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/'

full_data = pd.read_csv(project_path + 'nhanes_2013_2014_full_data.csv',index_col=0)

data_cols = list(full_data.columns)
bio_cols = []
que_cols = []

for col in data_cols:
    if ('LBX' in col):
        bio_cols.append(col)
    else:
        que_cols.append(col)

biochemistry_data = full_data[bio_cols]
questionnaire_data = full_data[que_cols]

if False:
    print(biochemistry_data.describe())
    print("\n---------------------------\n")
    print(questionnaire_data.describe())

if True:
    print(biochemistry_data.columns)
    print("\n---------------------------\n")
    print(questionnaire_data.columns)



pair_df = pd.concat([biochemistry_data.LBXSTR, questionnaire_data.HYPERTENSION], axis=1)
print(pair_df.describe())

cols = list(pair_df.columns)
corr_matrix = np.corrcoef(pair_df[cols].values.T)

print("\nCorrelation matrix:")
print(corr_matrix)

from scipy.stats import pearsonr
print("\nPearson's corr coeff:")
print(pearsonr(biochemistry_data.LBXSTR,questionnaire_data.HYPERTENSION))

from scipy.stats import pointbiserialr
print("\nPoint biserial corr coeff:")
print(pointbiserialr(biochemistry_data.LBXSTR,questionnaire_data.HYPERTENSION))