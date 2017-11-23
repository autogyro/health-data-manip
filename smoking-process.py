#Testing loading datasets
#@Juan E. Rolon
#https://github.com/juanerolon

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#path to datasets folders
datasets_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/'
project_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/'

#Import smoking behavior dataset
smoking_data = pd.read_sas(datasets_path + 'smoking/SMQ_H.XPT')

#Import alcohol-age cleaned dataset using Apache Arrow feather format
import feather
ager_alcohol_data = feather.read_dataframe(project_path + 'ager_alcohol_data.feather')
ager_alcohol_data = ager_alcohol_data.set_index('SEQNDX') #Restore the index saved during the export step


smoking_data = smoking_data[['SEQN','SMQ040']]
smoking_data = smoking_data.set_index('SEQN')
smoking_data = smoking_data.reindex(ager_alcohol_data.index)

smoking_data.dropna(axis=0, how='any', inplace=True)

print("Smoking behavior dataframe:\n")
print(smoking_data.head(5))
print(smoking_data.describe())

ager_alcohol_data = ager_alcohol_data.reindex(smoking_data.index)

print("Alcohol consumption consistent dataframe:\n")
print(ager_alcohol_data.head())
print(ager_alcohol_data.describe())

print("*"*70)
print("\nMerged dataframe\n")

merged_data = pd.concat([ager_alcohol_data, smoking_data])

print(merged_data.head())
print(merged_data.describe())


print("\nSMQ040 Code values: \n 1:Every day, 2:Some days, 3:Not at all, 7:Refused, 9:Don't know")

nSMQ040_YES_EVRY = merged_data[merged_data.SMQ040 == 1.0].SMQ040.count()
nSMQ040_YES_SOME = merged_data[merged_data.SMQ040 == 2.0].SMQ040.count()
nSMQ040_NO = merged_data[merged_data.SMQ040 == 3.0].SMQ040.count()
nSMQ040_unknown = merged_data[merged_data.SMQ040 == 9.0].SMQ040.count()
nSMQ040_refused = merged_data[merged_data.SMQ040 == 7.0].SMQ040.count()

print("Number of records SMQ040  containing YES EVERY DAY answer: {}".format(nSMQ040_YES_EVRY))
print("Number of records SMQ040  containing YES SOME DAYS answer: {}".format(nSMQ040_YES_SOME))
print("Number of records SMQ040  containing NO answer: {}".format(nSMQ040_NO))
print("Number of records SMQ040  containing refused answer: {}".format(nSMQ040_refused))
print("Number of records SMQ040  containing unknown answer: {}".format(nSMQ040_unknown))

#We will need to merge/encode SMQ040 =1 SMQ040 =2 into a singled True (YES or 1) value