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

#Process demographics dataframe
print("")
demographics_data = pd.read_sas(datasets_path + 'demographics/DEMO_H.XPT')

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
#Downcast selected features
ager_data['AGE'] = pd.to_numeric(ager_data['AGE'], downcast='integer')
ager_data['GENDER'] = pd.to_numeric(ager_data['GENDER'], downcast='integer')
ager_data['INCOME_LEVEL'] = pd.to_numeric(ager_data['INCOME_LEVEL'], downcast='integer')

print(ager_data.head())
print("\nCounts:")
print(ager_data.count())

#Process alcohol consumption dataframe
print("")
alcohol_data = pd.read_sas(datasets_path + 'alcohol_use/ALQ_H.XPT')
alcohol_data = txt.switch_df_index(alcohol_data, 'SEQN')
alcohol_data = alcohol_data[['ALQ130']]
alcohol_data.rename(columns = {'ALQ130':'ALCOHOL_NUM'}, inplace=True)
alcohol_data['ALCOHOL_NUM'].fillna(value=0, inplace=True)
alcohol_data['ALCOHOL_NUM'] = pd.to_numeric(alcohol_data['ALCOHOL_NUM'], downcast='integer')
alcohol_data.ALCOHOL_NUM.replace(to_replace=999, value=0, inplace=True)

print(alcohol_data.head())
print("\nCounts:")
print(alcohol_data.count())


