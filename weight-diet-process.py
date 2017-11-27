#Testing loading datasets
#@Juan E. Rolon
#https://github.com/juanerolon

import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#path to datasets folders
datasets_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/'
project_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/'

#Import weight and nutrition dataset
weight_data = pd.read_sas(datasets_path + 'weight_history/WHQ_H.XPT')
nutrition_data = pd.read_sas(datasets_path + 'diet_nutrition/DBQ_H.XPT')

#Import alcohol-age cleaned dataset using Apache Arrow feather format
import feather
alcohol_smoking_data = feather.read_dataframe(project_path + 'alcohol_smoking_data.feather')
alcohol_smoking_data = alcohol_smoking_data.set_index('SEQNDX') #Restore the index saved during the export step

weight_data = weight_data[['SEQN', 'WHD010', 'WHD020']]
#Purge rows with NaNs
weight_data.dropna(axis=0, how='any', inplace=True)

print(weight_data.describe())
#WHD010 = 7777 Refused
#WHD010 = 9999 Unknown
#WHD020 = 7777 Refused
#WHD020 = 9999 Unknown
print("WHD010 - Current self-reported height (inches)")
print("WHD020 - Current self-reported weight (pounds)")

nWHD010_refused = weight_data[weight_data.WHD010 == 7777.0].WHD010.count()
nWHD010_unknown = weight_data[weight_data.WHD010 == 9999.0].WHD010.count()

nWHD020_refused = weight_data[weight_data.WHD020 == 7777.0].WHD020.count()
nWHD020_unknown = weight_data[weight_data.WHD020 == 9999.0].WHD020.count()

print("WHD010 refused answers: {}".format(nWHD010_refused))
print("WHD010 unknown answers: {}".format(nWHD010_unknown))
print("WHD020 refused answers: {}".format(nWHD020_refused))
print("WHD020 unknown answers: {}".format(nWHD020_unknown))

#Purge rows containing unknown or refused answers in variables WHD010 and WHD020
print("Purging rows with refused and unknown answers......\n")
weight_data = weight_data[weight_data.WHD010 != 7777.0]
weight_data = weight_data[weight_data.WHD010 != 9999.0]
weight_data = weight_data[weight_data.WHD020 != 7777.0]
weight_data = weight_data[weight_data.WHD020 != 9999.0]

#simple BMI formula
def bmi(h, w):
    return np.round((w/h**2) * 703.0, 2)
print("Adding BMI column .....\n")
weight_data['BMI'] = bmi(weight_data['WHD010'], weight_data['WHD020'])

#Downcast SEQN to ints
weight_data['SEQN'] = pd.to_numeric(weight_data['SEQN'], downcast='integer')
weight_data = weight_data.set_index('SEQN')
weight_data = weight_data.reindex(alcohol_smoking_data.index)
#Purge rows with NaNs
weight_data.dropna(axis=0, how='any', inplace=True)

#print(weight_data.head())
#print(weight_data.describe())

alcohol_smoking_data = alcohol_smoking_data.reindex(weight_data.index)
alcohol_smoking_data.dropna(axis=0, how='any', inplace=True)

merged_data = alcohol_smoking_data.copy()
merged_data['BMI'] = weight_data['BMI']

#We need at this point to rename some columns
#for better readability:



print(merged_data.head())
print(merged_data.describe())


#----- Nutrition data ----------#
print("*"*80 +"\nNutrition data preview: \n")
nutrition_data = np.round(nutrition_data[['SEQN','DBD895', 'DBD900', 'DBD905', 'DBD910']],2)
nutrition_data .dropna(axis=0, how='any', inplace=True)

#downcast SEQN to integer values and set it as dataframe index
nutrition_data['SEQN'] = pd.to_numeric(nutrition_data['SEQN'], downcast='integer')
nutrition_data = nutrition_data.set_index('SEQN')

codes = \
"\nDBD895 Num. of meals not home prepared in past 7 days (Range:0 to 21)\n" +\
"DBD900 Num, of meals from fast food or pizza place past 7 days (Range:0 to 21)\n" +\
"DBD905 Num. of ready-to-eat foods in past 30 days (Range:0 to 180)\n" +\
"DBD910 Num. of frozen meals/pizza in past 30 days (Range:0 to 180)\n"
print(codes)

#DBD895 = 5555 More than 21
#DBD895 = 7777 Refused
#DBD895 = 9999 Unknown

#DBD900 = 5555 More than 21
#DBD900 = 7777 Refused
#DBD900 = 9999 Unknown

#Purge rows with refused or unknown answers
nutrition_data = nutrition_data[nutrition_data.DBD895  != 7777.0]
nutrition_data = nutrition_data[nutrition_data.DBD895  != 9999.0]

nutrition_data = nutrition_data[nutrition_data.DBD900 != 7777.0]
nutrition_data = nutrition_data[nutrition_data.DBD900 != 9999.0]

nutrition_data = nutrition_data[nutrition_data.DBD905 != 7777.0]
nutrition_data = nutrition_data[nutrition_data.DBD905 != 9999.0]

nutrition_data = nutrition_data[nutrition_data.DBD910 != 7777.0]
nutrition_data = nutrition_data[nutrition_data.DBD910 != 9999.0]

print("DBD895, DBD900 > 21 values:\n")
print(nutrition_data[nutrition_data.DBD895  > 21.0].DBD895)
print(nutrition_data[nutrition_data.DBD900  > 21.0].DBD900)
print("........ \n")

#Also purge unreasonable values of weekly junk food and/or pizza consumption
nutrition_data = nutrition_data[nutrition_data.DBD895  != 5555.0]
nutrition_data = nutrition_data[nutrition_data.DBD895  != 5555.0]

#Align nutrition data with merged data from previous step
nutrition_data = nutrition_data.reindex(merged_data.index)
nutrition_data .dropna(axis=0, how='any', inplace=True)

#define function to transform food consumption to percentage values
#based on an equivalence of 100% to 21 for DBD895, DBD900, and
#100% to 180 for DBD905, DBD910, respectivly.
def percent_transform(x,n):
    return np.round(float(x/n *100.0),2)


nutrition_data['7-Day NotHome'] = nutrition_data['DBD895'].apply(lambda x: percent_transform(x, 21))
nutrition_data['7-Day FastFood'] = nutrition_data['DBD900'].apply(lambda x: percent_transform(x, 21))
nutrition_data['30-Day ReadyFood'] = nutrition_data['DBD905'].apply(lambda x: percent_transform(x, 180))
nutrition_data['30-Day FrozenFood'] = nutrition_data['DBD910'].apply(lambda x: percent_transform(x, 180))

print(nutrition_data.head())
print(np.round(nutrition_data.describe(), 2))