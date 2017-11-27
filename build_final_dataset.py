# Testing loading datasets
# @Juan E. Rolon
# https://github.com/juanerolon

import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot


#Path to datasets folders
datasets_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/'
project_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/'

#Datasets
insurance_data = pd.read_sas(datasets_path + 'health_insurance/HIQ_H.xpt')
cholpressure_data = pd.read_sas(datasets_path + 'blood_pressure/BPQ_H.XPT')
cardiovascular_data = pd.read_sas(datasets_path + 'cardiovascular/CDQ_H.XPT')
diabetes_data = pd.read_sas(datasets_path + 'diabetes/DIQ_H.XPT')

#Import alcohol-age cleaned dataset using Apache Arrow feather format
import feather
merged_data = feather.read_dataframe(project_path + 'independent_variables_df.feather')
merged_data = merged_data.set_index('SEQN') #Restore the index saved during the export step

print("\nMerged Data Previews Initial Stage:\n")
print(merged_data.head())
print(merged_data.describe())

#Process insurance data
print("Processing insurance data ......\n")

insurance_data = insurance_data[['SEQN', 'HIQ011']]
insurance_data['SEQN'] = pd.to_numeric(insurance_data['SEQN'], downcast='integer')
insurance_data = insurance_data.set_index('SEQN')
insurance_data = insurance_data.reindex(merged_data.index)

insurance_data.dropna(axis=0, how='any', inplace=True)

insurance_data = insurance_data[insurance_data.HIQ011 != 7.0] #Purge records with refused answers
insurance_data = insurance_data[insurance_data.HIQ011 != 9.0] #Purge records with unknown answers


def binarize_ins(x):
    if (x == 1.0):
        return 1
    elif (x== 2.0):
        return 0
    else:
        return np.nan

#Rename 'HIQ011' -- > 'Insurance' and binarize the corresponding record values
insurance_data.rename(columns = {'HIQ011':'Insurance'}, inplace=True)
insurance_data['Insurance'] = insurance_data['Insurance'].apply(lambda x: binarize_ins(x))

merged_data['Insurance'] = insurance_data['Insurance']

print("\nMerged Data Previews Second Stage:\n")
print(merged_data.head())
print(merged_data.describe())

#------------------------------------------------------------------------------------------
print("\n\nProcesing cholesterol and hypertension data......\n\n")
cholpressure_data = cholpressure_data[['SEQN', 'BPQ020','BPQ040A','BPQ080','BPQ090D']]
cholpressure_data.rename(columns = {'BPQ020':'Hypertension'}, inplace=True)
cholpressure_data.rename(columns = {'BPQ040A':'HypertenMED'}, inplace=True)
cholpressure_data.rename(columns = {'BPQ080':'HighCholesterol'}, inplace=True)
cholpressure_data.rename(columns = {'BPQ090D':'HighCholMED'}, inplace=True)

cholpressure_data['SEQN'] = pd.to_numeric(cholpressure_data['SEQN'], downcast='integer')
cholpressure_data = cholpressure_data.set_index('SEQN')

cholpressure_data = cholpressure_data.reindex(merged_data.index)

cholpressure_data.fillna(value=0, inplace=True)

print(cholpressure_data.head())
print(cholpressure_data.describe())

for col in cholpressure_data.columns:
    print(col)