
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


print("\nPostive Cases\n")
#@@@@@@@@@@@@@@ Positive cases
iron_data_positive_chest = iron_data_a[iron_data_a['CHEST_PAIN_30MIN'] == 1]
if False: txt.headcounts(iron_data_positive_chest)
print(iron_data_positive_chest['LBXSIR'].describe())

iron_data_positive_hyp = iron_data_a[iron_data_a['HYPERTENSION'] == 1]
if False: txt.headcounts(iron_data_positive_hyp)
print(iron_data_positive_hyp['LBXSIR'].describe())

iron_data_positive_chol = iron_data_a[iron_data_a['HIGHCHOL'] == 1]
if False: txt.headcounts(iron_data_positive_chol)
print(iron_data_positive_chol['LBXSIR'].describe())

print("\nNegative Cases\n")
#@@@@@@@@@@@@@@ Negative cases
iron_data_negative_chest = iron_data_a[iron_data_a['CHEST_PAIN_30MIN'] == 0]
if False: txt.headcounts(iron_data_negative_chest)
print(iron_data_negative_chest['LBXSIR'].describe())

iron_data_negative_hyp = iron_data_a[iron_data_a['HYPERTENSION'] == 0]
if False: txt.headcounts(iron_data_negative_hyp)
print(iron_data_negative_hyp['LBXSIR'].describe())

iron_data_negative_chol = iron_data_a[iron_data_a['HIGHCHOL'] == 0]
if False: txt.headcounts(iron_data_negative_chol)
print(iron_data_negative_chol['LBXSIR'].describe())

