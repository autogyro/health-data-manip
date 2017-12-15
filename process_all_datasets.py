# Script will load and process all datasets in selected NHANES surveys
# @Juan E. Rolon
# https://github.com/juanerolon

import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot

#import local modules
from importlib.machinery import SourceFileLoader
mod_path = "/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/nhanes2013-2104/"
txt = SourceFileLoader("text_utils", mod_path+"text_utils.py").load_module()
gut = SourceFileLoader("graph_utils", mod_path+"graph_utils.py").load_module()


#Path to datasets folders

datasets_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/'
project_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/'

######## Load datasets ########

#Demographics dataset
demographics_data = pd.read_sas(datasets_path + 'demographics/DEMO_H.XPT')

#Alcohol consumption dataset
alcohol_data = pd.read_sas(datasets_path + 'alcohol_use/ALQ_H.XPT')

###############################

#Process demographics dataframe

ager_data = txt.restrict_by_interval(demographics_data, 'RIDAGEYR', 18, 45, 'inclusive')
ager_data = txt.switch_df_index(ager_data, 'SEQN')
#Restrict dataframe to selected columns
ager_data = ager_data[['RIDAGEYR', 'RIAGENDR','INDHHIN2']]
#Rename original columns to user specified names
ager_data.rename(columns = {'RIDAGEYR':'AGE', 'RIAGENDR':'GENDER', 'INDHHIN2':'INCOME_LEVEL'}, inplace=True)
#Remove records (rows) with NaNs, and those containing 77.0 ('Refused') or 99 ('Unknown')
ager_data.dropna(axis=0, how='any', inplace=True)
ager_data = ager_data[ager_data.INCOME_LEVEL != 77.0]
ager_data = ager_data[ager_data.INCOME_LEVEL != 99.0]

print(ager_data.head())


#Process alcohol consumption dataframe





