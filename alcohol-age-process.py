#Testing loading datasets
#@Juan E. Rolon
#https://github.com/juanerolon


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#path to datasets folders
datasets_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/'

#datafiles are in SAS format
demographics_data = pd.read_sas(datasets_path + 'demographics/DEMO_H.XPT')

age_g18 = demographics_data[(demographics_data.RIDAGEYR >= 18.0)]
ager_data = age_g18[(age_g18.RIDAGEYR <= 45.0)]

nag = ager_data.RIDAGEYR.count()
print("Num records of age-restricted data (18 <= age <= 45): {}".format(nag))

age_group_seqns = list(ager_data.SEQN)
print("Num seqn numbers of age-restricted data: {} ".format(len(age_group_seqns)))

#-------------------------------------------------------------------
#switch index of dataframe to be the unique seqn numbers
ager_data = ager_data.set_index('SEQN')

nvals_age = ager_data.RIDAGEYR.count()
nansv_age = ager_data.RIDAGEYR.isnull().sum()

nvals_gender = ager_data.RIAGENDR.count()
nansv_gender = ager_data.RIAGENDR.isnull().sum()

nvals_income = ager_data.INDHHIN2.count()
nansv_income = ager_data.INDHHIN2.isnull().sum()

print("\nNum of values in RIDAGEYR column: {}".format(nvals_age))
print("Num of NaNs   in RIDAGEYR column: {}\n".format(nansv_age))

print("\nNum of values in RIAGENDR column: {}".format(nvals_gender))
print("Num of NaNs   in RIAGENDR column: {}\n".format(nansv_gender))

print("\nNum of values in INDHHIN2 column: {}".format(nvals_income))
print("Num of NaNs   in INDHHIN2 column: {}\n".format(nansv_income))
#--------------------------------------------------------------------

alcohol_data = pd.read_sas(datasets_path + 'alcohol_use/ALQ_H.XPT')
alcohol_data = alcohol_data.set_index('SEQN')
alcohol_data = alcohol_data.reindex(ager_data.index)

nalch = len(list(alcohol_data.index))
nvals101 = alcohol_data.ALQ101.count()
nansv101 = alcohol_data.ALQ101.isnull().sum()

nvals130 = alcohol_data.ALQ130.count()
nansv130 = alcohol_data.ALQ130.isnull().sum()

print("\nNum records in age-restricted alcohol data: {}".format(nalch))
print("Num of values in ALQ101 column: {}".format(nvals101))
print("Num of NaNs   in ALQ101 column: {}\n".format(nansv101))

print("Num of values in ALQ130 column: {}".format(nvals130))
print("Num of NaNs   in ALQ130 column: {}\n".format(nansv130))

#-----------------------------------------------------------------------
#Revisiting income data
print("\nRevisiting income data:\n")
print(ager_data.INDHHIN2.head(10))
print(ager_data.INDHHIN2.describe())
