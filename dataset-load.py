#Testing loading ASCII/TSV Files
#@Juan E. Rolon

import pandas as pd

demographics_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/demographics/'
smoking_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/smoking/'


#datafiles are in SAS format
demographics_data = pd.read_sas(demographics_path + 'DEMO_H.XPT')
smoking_data = pd.read_sas(smoking_path + 'SMQ_H.XPT')

#datasets keys or column fields
demographics_vars = list(demographics_data.keys())
smoking_vars = list(smoking_data.keys())

#dataset quick preview of a subset of the column fields (variables of interest)
print("\nDemographics data exploration:\n")
print(demographics_data[['SEQN', 'RIDAGEYR', 'RIAGENDR', 'INDHHIN2']].head())

print("\nSmoking cigarettes use data exploration:\n")
print(smoking_data[['SMQ040', 'SMQ020']].head())
