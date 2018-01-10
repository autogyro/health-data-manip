
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

full_data = pd.read_csv(project_path + 'full_data_raw.csv',index_col=0)
if False: txt.headcounts(full_data)

iron_data_a = full_data[['HYPERTENSION_ONSET', 'HIGHCHOL_ONSET', 'HIGHCHOL', 'HYPERTENSION', 'CHEST_DISCOMFORT', 'CHEST_PAIN_30MIN','LBXSIR']]
if False: txt.headcounts(iron_data_a)


feat_list = ['CHEST_PAIN_30MIN', 'HYPERTENSION', 'HIGHCHOL']
target = 'LBXSIR'
cases = ['Negative cases', 'Positive cases']

for m, case in enumerate(cases):
    print(case)
    for feat in feat_list:
        df = full_data[full_data[feat]==m]
        print(feat)
        print(df[target].describe())








