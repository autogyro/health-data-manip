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
demographics_data = txt.switch_df_index(demographics_data, 'SEQN')
demographics_data = demographics_data[['RIDAGEYR', 'RIAGENDR','RIDRETH3','INDHHIN2']]
#Restrict dataframe to selected columns
demographics_data = txt.restrict_by_interval(demographics_data, 'RIDAGEYR', 18, 65, 'inclusive')
#Rename original columns to user specified names
demographics_data.rename(columns = {'RIDAGEYR':'AGE', 'RIAGENDR':'GENDER',
                            'RIDRETH3':'ETHNICITY','INDHHIN2':'INCOME_LEVEL'}, inplace=True)

demo_features = ['AGE','GENDER','ETHNICITY','INCOME_LEVEL']

demographics_data.INCOME_LEVEL.fillna(value=7, inplace=True)
for val in [77.0,99.0]:
    demographics_data['INCOME_LEVEL'].replace(to_replace=val, value=7, inplace=True)
#Downcast selected features
for feat in demo_features:
    demographics_data[feat] = pd.to_numeric(demographics_data[feat], downcast='integer')


txt.headcounts(demographics_data)

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

alcohol_features = ['ALCOHOL_NUM']

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

smoking_features = ['SMOKING']

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

weight_features = ['BMI']

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

nutrition_features = ['NOTHOME_FOOD', 'FAST_FOOD']

txt.headcounts(nutrition_data)


#Process nutrition data
#######################
insurance_data = pd.read_sas(datasets_path + 'health_insurance/HIQ_H.xpt')
insurance_data = txt.switch_df_index(insurance_data, 'SEQN')
insurance_data = insurance_data[['HIQ011']]
insurance_data.rename(columns = {'HIQ011':'INSURANCE'}, inplace=True)
insurance_data['INSURANCE'].fillna(value=0, inplace=True)
insurance_data['INSURANCE'] = pd.to_numeric(insurance_data['INSURANCE'], downcast='integer')
for val in [2,7,9]:
    insurance_data.INSURANCE.replace(to_replace=val, value=0, inplace=True)

insurance_features = ['INSURANCE']

txt.headcounts(insurance_data)
txt.get_feature_counts(insurance_data, ['INSURANCE'])
txt.count_feature_nans(insurance_data, ['INSURANCE'])


#Process blood-pressure cholesterol text data
#############################################

cholpressure_data = pd.read_sas(datasets_path + 'blood_pressure/BPQ_H.XPT')
cholpressure_data = txt.switch_df_index(cholpressure_data, 'SEQN')
cholpressure_data = cholpressure_data[['BPQ020','BPQ030', 'BPQ040A', 'BPQ050A', 'BPQ080', 'BPQ090D']]

old_names = ['BPQ020','BPQ030', 'BPQ040A', 'BPQ050A', 'BPQ080', 'BPQ090D']
cholpressure_features = ['HYPERTENSION_ONSET', 'HYPERTENSION_1', 'HYPERTENSION_2', 'HYPERTENSION_3', 'HIGHCHOL_ONSET', 'HIGHCHOL']

for n, new_name in enumerate(cholpressure_features):
    cholpressure_data.rename(columns={old_names[n]: new_name}, inplace=True)

#------global nans imputation
cholpressure_data.fillna(value=0, inplace=True)

for feat in cholpressure_features:
    cholpressure_data[feat] = pd.to_numeric(cholpressure_data[feat], downcast='integer')
#-----missing values imputation
for val in [2,7,9]:
    cholpressure_data.replace(to_replace=val, value=0, inplace=True)
#-----combine HYPERTENSION features
cholpressure_data['HYPERTENSION'] = np.vectorize(txt.bit_logic)\
    (cholpressure_data['HYPERTENSION_1'].values, cholpressure_data['HYPERTENSION_2'].values, 'OR')

cholpressure_data['HYPERTENSION'] = np.vectorize(txt.bit_logic)\
    (cholpressure_data['HYPERTENSION'].values, cholpressure_data['HYPERTENSION_3'].values, 'OR')

cholpressure_data.drop(['HYPERTENSION_1', 'HYPERTENSION_2', 'HYPERTENSION_3'], axis=1, inplace=True)

cholpressure_features = ['HYPERTENSION_ONSET', 'HYPERTENSION', 'HIGHCHOL_ONSET', 'HIGHCHOL']

txt.headcounts(cholpressure_data)
txt.get_feature_counts(cholpressure_data, cholpressure_features)


#Process cardiovascular text data
#############################################
cardiovascular_data = pd.read_sas(datasets_path + 'cardiovascular/CDQ_H.XPT')  #DONE
cardiovascular_data = txt.switch_df_index(cardiovascular_data, 'SEQN')
cardiovascular_data = cardiovascular_data[['CDQ001', 'CDQ008', 'CDQ010']]
cardiovascular_data.rename(columns = {'CDQ001':'CHEST_DISCOMFORT'}, inplace=True)
cardiovascular_data.rename(columns = {'CDQ008':'CHEST_PAIN_30MIN'}, inplace=True)
cardiovascular_data.rename(columns = {'CDQ010':'BREATH_SHORTNESS'}, inplace=True)

cardio_features = ['CHEST_DISCOMFORT', 'CHEST_PAIN_30MIN', 'BREATH_SHORTNESS']

cardiovascular_data.fillna(value=0, inplace=True)
for feat in cardio_features:
    cardiovascular_data[feat] = pd.to_numeric(cardiovascular_data[feat], downcast='integer')
for val in [2,7,9]:
    cardiovascular_data.replace(to_replace=val, value=0, inplace=True)


txt.headcounts(cardiovascular_data)



#Process diabetes text data
#############################################
diabetes_data = pd.read_sas(datasets_path + 'diabetes/DIQ_H.XPT') #DONE
diabetes_data = txt.switch_df_index(diabetes_data, 'SEQN')
diabetes_data = diabetes_data[['DIQ175A','DIQ010', 'DIQ160', 'DIQ170']]
old_diab_features = ['DIQ175A','DIQ010', 'DIQ160', 'DIQ170']

diabetes_features = ['FAMILIAL_DIABETES', 'DIAGNOSED_DIABETES', 'DIAGNOSED_PREDIABETES', 'RISK_DIABETES']

for n, new_name in enumerate(diabetes_features):
    diabetes_data.rename(columns={old_diab_features[n]: new_name}, inplace=True)
diabetes_data.fillna(value=0, inplace=True)
for feat in diabetes_features:
    diabetes_data[feat] = pd.to_numeric(diabetes_data[feat], downcast='integer')

diabetes_data['FAMILIAL_DIABETES'].replace(to_replace=10, value=1, inplace=True)
for val in [77,99]:
    diabetes_data['FAMILIAL_DIABETES'].replace(to_replace=val, value=0, inplace=True)

for val in [1,3]:
    diabetes_data['DIAGNOSED_DIABETES'].replace(to_replace=val, value=1, inplace=True)
for val in [2,7,9]:
    diabetes_data['DIAGNOSED_DIABETES'].replace(to_replace=val, value=0, inplace=True)

for val in [2,7,9]:
    diabetes_data['DIAGNOSED_PREDIABETES'].replace(to_replace=val, value=0, inplace=True)
    diabetes_data['RISK_DIABETES'].replace(to_replace=val, value=0, inplace=True)

txt.headcounts(diabetes_data)


#REINDEX DATAFRAMES TO CONFORM TO DEMOGRAPHICS DATA SEQN INDEX
##############################################################

#Process alcohol consumption dataframe
#-------------------------------------
print("\nReindex alcohol dataframe using indexes from demographics dataframe")
txt.bitwise_index_compare(demographics_data,alcohol_data) # compare indexes respect to master dataframe (demographics)
alcohol_data = alcohol_data.reindex(demographics_data.index) # reindex
mean_alcohol = int(np.round(alcohol_data.mean())) #compute mean values of features
alcohol_data.fillna(value=mean_alcohol, inplace=True) # impute mean values replacing nans

txt.count_feature_nans(alcohol_data, alcohol_features)
txt.headcounts(alcohol_data)

# print(alcohol_data.loc[index_set_1])
# print(demographics_data.AGE.loc[index_set_1])

#Process smoking consumption dataframe
#-------------------------------------
print("\nReindex smoking dataframe using indexes from alcohol dataframe")
txt.bitwise_index_compare(alcohol_data, smoking_data)
smoking_data = smoking_data.reindex(alcohol_data.index)
txt.count_feature_nans(smoking_data, smoking_features)
txt.headcounts(smoking_data)

#Process weight history dataframe
#-------------------------------------
print("\nReindex weight dataframe using indexes from smoking dataframe")
txt.bitwise_index_compare(smoking_data,weight_data)
weight_data = weight_data.reindex(smoking_data.index)
txt.count_feature_nans(weight_data,weight_features)
mean_bmi = int(np.round(weight_data['BMI'].mean())) #compute mean values of features
weight_data.fillna(value=mean_bmi, inplace=True)
txt.count_feature_nans(weight_data,weight_features)
txt.headcounts(weight_data,2)


#Process nutrition dataframe
#-------------------------------------
print("\nReindex nutrition dataframe using indexes from weight history dataframe")
txt.bitwise_index_compare(weight_data,nutrition_data)

nutrition_data = nutrition_data.reindex(weight_data.index)

txt.count_feature_nans(nutrition_data, nutrition_features)

txt.headcounts(nutrition_data)

for feat in nutrition_features:
    mval = np.round(nutrition_data[feat].mean(),2)
    nutrition_data[feat].fillna(value=mval,inplace=True)
    print(mval)

txt.headcounts(nutrition_data)
txt.count_feature_nans(nutrition_data, nutrition_features)

#Process blood-pressure cholesterol dataframe
#--------------------------------------------

print("\nReindex cholesterol blood-pressure dataframe using indexes from nutrition dataframe")
txt.bitwise_index_compare(nutrition_data,cholpressure_data)
cholpressure_data = cholpressure_data.reindex(nutrition_data.index)
txt.headcounts(cholpressure_data)
txt.count_feature_nans(cholpressure_data, cholpressure_features)

#Process cardiovascular dataframe
#--------------------------------------------
print("\nReindex cardiovascular dataframe using indexes from cholpressure dataframe")
txt.bitwise_index_compare(cholpressure_data,cardiovascular_data)
cardiovascular_data = cardiovascular_data.reindex(cholpressure_data.index)

txt.headcounts(cardiovascular_data)
txt.count_feature_nans(cardiovascular_data, cardio_features)

for feat in cardio_features:
    val = 0
    cardiovascular_data[feat].fillna(value=val,inplace=True)

txt.headcounts(cardiovascular_data)
txt.count_feature_nans(cardiovascular_data, cardio_features)

#Process diabetes dataframe
#--------------------------------------------
print("\nReindex diabetes dataframe using indexes from cardiovascular dataframe")
txt.bitwise_index_compare(cardiovascular_data, diabetes_data)
diabetes_data = diabetes_data.reindex(demographics_data.index)
txt.count_feature_nans(diabetes_data, diabetes_features)


#CONCATENATE ALL DATAFRAMES
##############################################################

nhanes_datasets_part_1 = [demographics_data, alcohol_data, smoking_data, weight_data, nutrition_data, cholpressure_data,
                         cardiovascular_data, diabetes_data]

nhanes_df1 = pd.concat(nhanes_datasets_part_1, axis=1)


txt.headcounts(nhanes_df1)
txt.count_feature_nans(nhanes_df1, list(nhanes_df1.columns))