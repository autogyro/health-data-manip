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

# print("\nMerged Data Previews Fifth Stage:\n")
# print(merged_data.head())
# print(merged_data.describe())


#Save Fifth stage merged dataframe
filename_pre = 'prefinal_df.feather'
merged_data['SEQN'] = merged_data.index
feather.write_dataframe(merged_data, filename_pre)

#---------------------------------------- Further processing ----------------------------------------------------------
#******* Categorizing functions *********

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

print("\nCheck alcohol consumption frequency counts:\n")
alq_unique = merged_data['Alcohol'].unique()
snv =0
for val in np.sort(alq_unique):
    nv = merged_data[merged_data.Alcohol == val].Alcohol.count()
    snv += nv
    print("No. Drinks = {} ..... Count: {}".format(val, nv))
print('Count total: {}'.format(snv))

def categorize_alcohol(x):
    if x == 0:
        return 'Non-Drinker'
    elif ((x >=1) and (x <=2)):
        return 'ModerateDrinker'
    elif ((x > 2) and (x <= 5)):
        return 'Moderate-Heavy'
    elif ((x > 5) and (x <= 8)):
        return 'HeavyDrinker'
    elif (x > 8):
        return 'ExtremeDrinker'

def numcategorize_alcohol(x):
    if x == 0:
        return 0
    elif ((x >=1) and (x <=2)):
        return 1
    elif ((x > 2) and (x <= 5)):
        return 2
    elif ((x > 5) and (x <= 8)):
        return 3
    elif (x > 8):
        return 4

merged_data['AlcoholBracket'] = merged_data['Alcohol'].apply(lambda x: categorize_alcohol(x))
merged_data['NUM_AlcoholBracket'] = merged_data['Alcohol'].apply(lambda x: numcategorize_alcohol(x))

alq_unique = merged_data['AlcoholBracket'].unique()
snv = 0
for val in np.sort(alq_unique):
    nv = merged_data[merged_data.AlcoholBracket == val].AlcoholBracket.count()
    snv += nv
    print("Bracket = {} ..... Count: {}".format(val, nv))
print('Count total: {}'.format(snv))

# print("\nMerged Data Previews Sixth Stage:\n")
# print(merged_data.head())
# print(merged_data.describe())

if False:
    print("\nFast food consumption percentage")
    fast_food_unique = np.sort(merged_data['7-Day FastFood'].unique())
    snv = 0
    for val in np.sort(fast_food_unique):
        nv = merged_data[merged_data['7-Day FastFood'] == val]['7-Day FastFood'].count()
        snv += nv
        print("Percentage fast food = {} ..... Count: {}".format(val, nv))
    print('Count total: {}\n'.format(snv))

    print("^"*80)

    print("\nNot home food consumption percentage")
    nothome_food_unique = np.sort(merged_data['7-Day NotHome'].unique())
    snv = 0
    for val in np.sort(nothome_food_unique):
        nv = merged_data[merged_data['7-Day NotHome'] == val]['7-Day NotHome'].count()
        snv += nv
        print("Percentage not home = {} ..... Count: {}".format(val, nv))
    print('Count total: {}\n'.format(snv))

def categorize_food_cons(x):
    if (x >= 0.0) and (x <=15.0):
        return '0-15%'
    elif ((x > 15.0) and (x <= 25.0)):
        return '15-25%'
    elif ((x > 25.0) and (x <= 40.0)):
        return '25-40%'
    elif ((x > 40.0) and (x <= 60.0)):
        return '40-60%'
    elif (x > 60.0):
        return '60-100%'

def numcategorize_food_cons(x):
    if (x >= 0.0) and (x <=15.0):
        return 15
    elif ((x > 15.0) and (x <= 25.0)):
        return 25
    elif ((x > 25.0) and (x <= 40.0)):
        return 40
    elif ((x > 40.0) and (x <= 60.0)):
        return 60
    elif (x > 60.0):
        return 100

merged_data['NotHomeFood'] = merged_data['7-Day NotHome'].apply(lambda x: categorize_food_cons(x))
merged_data['FastFood'] = merged_data['7-Day FastFood'].apply(lambda x: categorize_food_cons(x))

merged_data['NUM_NotHomeFood'] = merged_data['7-Day NotHome'].apply(lambda x: numcategorize_food_cons(x))
merged_data['NUM_FastFood'] = merged_data['7-Day FastFood'].apply(lambda x: numcategorize_food_cons(x))

if False:
    print("\nNot home food consumption brackets")
    nothome_food_unique = np.sort(merged_data['NotHomeFood'].unique())
    snv = 0
    for val in np.sort(nothome_food_unique):
        nv = merged_data[merged_data['NotHomeFood'] == val]['NotHomeFood'].count()
        snv += nv
        print("Percentage Bracket not home = {} ..... Count: {}".format(val, nv))
    print('Count total: {}\n'.format(snv))

    print("^"*80)

    print("\nFast food consumption brackets")
    fast_food_unique = np.sort(merged_data['FastFood'].unique())
    snv = 0
    for val in np.sort(fast_food_unique):
        nv = merged_data[merged_data['FastFood'] == val]['FastFood'].count()
        snv += nv
        print("Percentage Bracket fast food = {} ..... Count: {}".format(val, nv))
    print('Count total: {}\n'.format(snv))


# print("\nMerged Data Previews Seventh Stage:\n")
# print(merged_data.head())
# # print(merged_data.describe())

def bmi_categorizer(x):
    if x < 18.5:
        return 'UnderWeight'
    elif (x >=18.5) and (x <=24.9):
        return 'HealthyWeight'
    elif (x > 24.9) and (x <= 29.9):
        return 'OeverWeight'
    elif x > 29.9:
        return 'Obese'

def numbmi_categorizer(x):
    if x < 18.5:
        return 1
    elif (x >=18.5) and (x <=24.9):
        return 2
    elif (x > 24.9) and (x <= 29.9):
        return 3
    elif x > 29.9:
        return 4

merged_data['BMI_BRACKET'] = merged_data['BMI'].apply(lambda x: bmi_categorizer(x))
merged_data['NUM_BMI_BRACKET'] = merged_data['BMI'].apply(lambda x: numbmi_categorizer(x))

if True:
    print("\nBMI Frequency Brackets")
    bmi_brackets_unique = np.sort(merged_data['BMI_BRACKET'].unique())
    snv = 0
    for val in np.sort(bmi_brackets_unique):
        nv = merged_data[merged_data['BMI_BRACKET'] == val]['BMI_BRACKET'].count()
        snv += nv
        print("BMI BRACKETS counts = {} ..... Count: {}".format(val, nv))
    print('Count total: {}\n'.format(snv))


print("\nMerged Data Previews Seventh Stage:\n")
print(merged_data.head())
print(merged_data.describe())

#Save Final stage merged dataframe
filename_final = 'final_health_survey_df.feather'
feather.write_dataframe(merged_data, filename_final)

csv_filename = 'final_health_survey_df.csv'
merged_data.to_csv(csv_filename)

print("\n\nEND Saving Final Dataframe")