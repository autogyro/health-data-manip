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
###############################
print("")
demographics_data = pd.read_sas(datasets_path + 'demographics/DEMO_H.XPT')

ager_data = txt.restrict_by_interval(demographics_data, 'RIDAGEYR', 18, 65, 'inclusive')
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

txt.headcounts(ager_data)

#Process alcohol consumption dataframe
######################################
print("")
alcohol_data = pd.read_sas(datasets_path + 'alcohol_use/ALQ_H.XPT')
alcohol_data = txt.switch_df_index(alcohol_data, 'SEQN')
alcohol_data = alcohol_data[['ALQ130']]
alcohol_data.rename(columns = {'ALQ130':'ALCOHOL_NUM'}, inplace=True)
alcohol_data['ALCOHOL_NUM'].fillna(value=0, inplace=True)
alcohol_data['ALCOHOL_NUM'] = pd.to_numeric(alcohol_data['ALCOHOL_NUM'], downcast='integer')
alcohol_data.ALCOHOL_NUM.replace(to_replace=999, value=0, inplace=True)

txt.headcounts(alcohol_data)

#Process smoking consumption dataframe
######################################
print("")
smoking_data = pd.read_sas(datasets_path + 'smoking/SMQ_H.XPT')
smoking_data = txt.switch_df_index(smoking_data, 'SEQN')
smoking_data = smoking_data[['SMQ040']]
smoking_data.rename(columns = {'SMQ040':'SMOKING'}, inplace=True)
smoking_data['SMOKING'].fillna(value=0, inplace=True)
smoking_data['SMOKING'] = pd.to_numeric(smoking_data['SMOKING'], downcast='integer')
smoking_data.SMOKING.replace(to_replace=1, value=1, inplace=True)
smoking_data.SMOKING.replace(to_replace=2, value=1, inplace=True)
smoking_data.SMOKING.replace(to_replace=0, value=0, inplace=True)
smoking_data.SMOKING.replace(to_replace=3, value=0, inplace=True)

txt.headcounts(smoking_data)

#Process weight data
####################
print("")
weight_data = pd.read_sas(datasets_path + 'weight_history/WHQ_H.XPT')
weight_data = txt.switch_df_index(weight_data, 'SEQN')
weight_data = weight_data[['WHD010', 'WHD020']]
weight_data.dropna(axis=0, how='any', inplace=True)

weight_data = weight_data[weight_data.WHD010 != 7777.0]
weight_data = weight_data[weight_data.WHD010 != 9999.0]
weight_data = weight_data[weight_data.WHD020 != 7777.0]
weight_data = weight_data[weight_data.WHD020 != 9999.0]

def bmi(h, w):
    return np.round((w/h**2) * 703.0, 2)

weight_data['BMI'] = bmi(weight_data['WHD010'], weight_data['WHD020'])
weight_data.drop(['WHD010', 'WHD020'], axis=1, inplace=True)

txt.headcounts(weight_data)


#Process nutrition data
#######################
nutrition_data = pd.read_sas(datasets_path + 'diet_nutrition/DBQ_H.XPT')
nutrition_data = txt.switch_df_index(nutrition_data, 'SEQN')
nutrition_data = nutrition_data[['DBD895', 'DBD900']]
nutrition_data.rename(columns = {'DBD895':'NOTHOME_FOOD', 'DBD900':'FAST_FOOD' }, inplace=True)
nutrition_data .dropna(axis=0, how='any', inplace=True)
nutrition_data['NOTHOME_FOOD'] = pd.to_numeric(nutrition_data['NOTHOME_FOOD'], downcast='integer')
nutrition_data['FAST_FOOD'] = pd.to_numeric(nutrition_data['FAST_FOOD'], downcast='integer')
#-----data imputation
mean_val1 = int(np.round(nutrition_data['NOTHOME_FOOD'].values.mean()))
nutrition_data.NOTHOME_FOOD.replace(to_replace=5555, value=mean_val1, inplace=True)
nutrition_data.NOTHOME_FOOD.replace(to_replace=9999, value=mean_val1, inplace=True)
mean_val2 = int(np.round(nutrition_data['FAST_FOOD'].values.mean()))
nutrition_data.FAST_FOOD.replace(to_replace=5555, value=mean_val2, inplace=True)
nutrition_data.FAST_FOOD.replace(to_replace=9999, value=mean_val2, inplace=True)

#-----data transformation to percentage values
def percent_transform(x,n):
    return np.round(float(x/n *100.0),2)

nutrition_data['NOTHOME_FOOD'] = nutrition_data['NOTHOME_FOOD'].apply(lambda x: percent_transform(x, 21))
nutrition_data['FAST_FOOD'] = nutrition_data['FAST_FOOD'].apply(lambda x: percent_transform(x, 21))

txt.headcounts(nutrition_data)


#Process nutrition data
#######################
insurance_data = pd.read_sas(datasets_path + 'health_insurance/HIQ_H.xpt')
insurance_data = txt.switch_df_index(insurance_data, 'SEQN')
insurance_data = insurance_data[['HIQ011']]

txt.headcounts(insurance_data)
txt.get_feature_counts(insurance_data, ['HIQ011'])


