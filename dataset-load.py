#Testing loading datasets
#@Juan E. Rolon

import pandas as pd

demographics_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/demographics/'
smoking_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/smoking/'
alcohol_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/alcohol_use/'
weight_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/weight_history/'
physical_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/physical_activity/'
pressurechol_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/blood_pressure/'


#datafiles are in SAS format
demographics_data = pd.read_sas(demographics_path + 'DEMO_H.XPT')
smoking_data = pd.read_sas(smoking_path + 'SMQ_H.XPT')
alcohol_data = pd.read_sas(alcohol_path + 'ALQ_H.XPT')
weight_data = pd.read_sas(weight_path + 'WHQ_H.XPT')
physical_data = pd.read_sas(physical_path + 'PAQ_H.XPT')
pressurechol_data = pd.read_sas(pressurechol_path + 'BPQ_H.XPT')

#datasets keys or column fields
demographics_vars = list(demographics_data.keys())
smoking_vars =  list(smoking_data.keys())
alcohol_vars = list(alcohol_data.keys())
weight_vars = list(alcohol_data.keys())
physical_vars = list(physical_data.keys())
pressurechol_vars = list(pressurechol_data.keys())

#datasets' quick preview of a subset of the column fields (variables of interest)
print("\nDemographics data exploration:\n")
print(demographics_data[['SEQN', 'RIDAGEYR', 'RIAGENDR', 'INDHHIN2']].head())

print("\nSmoking cigarettes use data exploration:\n")
print(smoking_data[['SEQN','SMQ040', 'SMQ020']].head())

print("\nAlcohol use data exploration:\n")
print(alcohol_data[['SEQN','ALQ120Q', 'ALQ130']].head())

print("\nWeight  history data exploration:\n")
print(weight_data[['SEQN','WHD010', 'WHD020']].head())

print("\nPhysical activity data exploration:\n")
print(physical_data[['SEQN','PAQ650', 'PAQ655']].head())

print("\nBlood pressure & cholesterol data exploration:\n")
print(pressurechol_data[['SEQN','BPQ020','BPQ040A', 'BPQ050A','BPQ080', 'BPQ090D']].head())
