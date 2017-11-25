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

print(weight_data.head())
print(weight_data.describe())


