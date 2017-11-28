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
insurance_data = pd.read_sas(datasets_path + 'health_insurance/HIQ_H.xpt')     #DONE
cholpressure_data = pd.read_sas(datasets_path + 'blood_pressure/BPQ_H.XPT')    #DONE
cardiovascular_data = pd.read_sas(datasets_path + 'cardiovascular/CDQ_H.XPT')  #DONE
diabetes_data = pd.read_sas(datasets_path + 'diabetes/DIQ_H.XPT') #DONE

#Import alcohol-age cleaned dataset using Apache Arrow feather format
import feather
merged_data = feather.read_dataframe(project_path + 'independent_variables_df.feather')
merged_data = merged_data.set_index('SEQN') #Restore the index saved during the export step

print("\nMerged Data Previews Initial Stage:\n")
print(merged_data.head())
print(merged_data.describe())

#-------------------------------- Process insurance data ----------------------------------------
#------------------------------------------------------------------------------------------------

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

merged_data['Insurance'].fillna(value=0, inplace=True) #Individuals to consuming alcohol (NaNs ---> 0)
merged_data['Insurance'] = pd.to_numeric(merged_data['Insurance'], downcast='integer')

print("\nMerged Data Previews Second Stage:\n")
print(merged_data.head())
print(merged_data.describe())

print("\nIncome levels")
print(merged_data.IncomeLevel.unique())


#-------------------------------- Process cholesterol, hypertension -----------------------------
#------------------------------------------------------------------------------------------------

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

def binarize_12(x):
    if (x == 1.0):
        return 1
    elif (x== 2.0):
        return 0
    else:
        return 0

#binarize
for col in cholpressure_data.columns:
    cholpressure_data[col] = cholpressure_data[col].apply(lambda x: binarize_12(x))

print(cholpressure_data.head())
print(cholpressure_data.describe())
print("")

for col in cholpressure_data.columns:
    print("Yes answers in column = {}".format(col))
    n = cholpressure_data[cholpressure_data[col] == 1][col].count()
    print(n)

#append all chol-pressure columns to merged_data dataframe
for col in cholpressure_data.columns:
    merged_data[col] = cholpressure_data[col]

print("\nMerged Data Previews Third Stage:\n")
print(merged_data.head())
print(merged_data.describe())


#-------------------------------- Process cardiovascular data -----------------------------------
#------------------------------------------------------------------------------------------------

print("\n\nProcessing cardiovascular data .......\n\n")
cardiovascular_data = cardiovascular_data[['SEQN', 'CDQ001', 'CDQ008', 'CDQ010']]
cardiovascular_data.rename(columns = {'CDQ001':'ChestDisc'}, inplace=True)
cardiovascular_data.rename(columns = {'CDQ008':'ChestPain'}, inplace=True)
cardiovascular_data.rename(columns = {'CDQ010':'ShortBreath'}, inplace=True)

cardiovascular_data['SEQN'] = pd.to_numeric(cardiovascular_data['SEQN'], downcast='integer')
cardiovascular_data = cardiovascular_data.set_index('SEQN')


cardiovascular_data = cardiovascular_data[cardiovascular_data.ChestDisc != 7.0] #Purge records with refused answers
cardiovascular_data = cardiovascular_data[cardiovascular_data.ChestDisc != 9.0] #Purge records with unknown answers

cardiovascular_data = cardiovascular_data[cardiovascular_data.ChestPain != 7.0] #Purge records with refused answers
cardiovascular_data = cardiovascular_data[cardiovascular_data.ChestPain != 9.0] #Purge records with unknown answers

cardiovascular_data = cardiovascular_data[cardiovascular_data.ShortBreath != 7.0] #Purge records with refused answers
cardiovascular_data = cardiovascular_data[cardiovascular_data.ShortBreath != 9.0] #Purge records with unknown answers


#binarize
for col in cardiovascular_data.columns:
    cardiovascular_data[col] = cardiovascular_data[col].apply(lambda x: binarize_12(x))

cardiovascular_data = cardiovascular_data.reindex(merged_data.index)
cardiovascular_data.fillna(value=0, inplace=True)

for col in cardiovascular_data.columns:
    cardiovascular_data[col] = pd.to_numeric(cardiovascular_data[col], downcast='integer')

print(cardiovascular_data.head())
print(cardiovascular_data.describe())

for col in cardiovascular_data.columns:
    n = cardiovascular_data[cardiovascular_data[col] == 1][col].count()
    print("Yes answers in column {} = {}: ".format(col,n))


#append all chol-pressure columns to merged_data dataframe
for col in cardiovascular_data.columns:
    merged_data[col] = cardiovascular_data[col]

print("\nMerged Data Previews Fourht Stage:\n")
print(merged_data.head())
print(merged_data.describe())

#-------------------------------- Process diabetes data ----------------------------------------
#-----------------------------------------------------------------------------------------------

print("\n\nProcessing diabetes data .......\n\n")



diabetes_data = diabetes_data[['SEQN', 'DIQ010', 'DIQ160', 'DIQ170']]

diabetes_data.rename(columns = {'DIQ010':'Diabetes'}, inplace=True)
diabetes_data.rename(columns = {'DIQ160':'Prediabetes'}, inplace=True)
diabetes_data.rename(columns = {'DIQ170':'RiskDiabetes'}, inplace=True)


diabetes_data['SEQN'] = pd.to_numeric(diabetes_data['SEQN'], downcast='integer')
diabetes_data = diabetes_data.set_index('SEQN')


for col in diabetes_data:
    diabetes_data = diabetes_data[diabetes_data[col] != 7.0] #Purge records with refused answers
    diabetes_data = diabetes_data[diabetes_data[col] != 9.0] #Purge records with unknown answers



diabetes_data.fillna(value=0, inplace=True)

def binarize_123(x):
    if (x == 1.0):
        return 1
    elif (x== 3.0):
        return 1
    elif (x== 2.0):
        return 0
    else:
        return 0

#binarize
for col in diabetes_data.columns:
    diabetes_data[col] = diabetes_data[col].apply(lambda x: binarize_123(x))

diabetes_data = diabetes_data.reindex(merged_data.index)
diabetes_data.fillna(value=0, inplace=True)

for col in diabetes_data.columns:
    diabetes_data[col] = pd.to_numeric(diabetes_data[col], downcast='integer')

print(diabetes_data.head())
print(diabetes_data.describe())

print("\n\nUnique values in columns:\n")
for col in diabetes_data:
    print(diabetes_data[col].unique())

for col in diabetes_data.columns:
    n = diabetes_data[diabetes_data[col] == 1][col].count()
    print("Yes answers in column {} = {}: ".format(col,n))

#append all chol-pressure columns to merged_data dataframe
for col in diabetes_data.columns:
    merged_data[col] = diabetes_data[col]

print("\nMerged Data Previews Fifth Stage:\n")
print(merged_data.head())
print(merged_data.describe())


#Save Fifth stage merged dataframe
filename_pre = 'prefinal_df.feather'
merged_data['SEQN'] = merged_data.index
feather.write_dataframe(merged_data, filename_pre)

#-------------- Further processing ------------------------------------
#Categorizing functions

#Gender categories ints to labels
def gender_cat(x):
    if x == 1:
        return 'Male'
    elif x == 2:
        return 'Female'
    else:
        return np.nan

merged_data['Gender'] = merged_data['Gender'].apply(lambda x: gender_cat(x))
print(merged_data.head())

#Check column unique values
if False:
    print("\n\nUnique values in columns:\n")
    for col in merged_data:
        print("Column {}:\n".format(col))
        print(merged_data[col].unique())