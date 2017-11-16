#Testing loading datasets
#@Juan E. Rolon

import pandas as pd

datasets_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/'

#datafiles are in SAS format
demographics_data = pd.read_sas(datasets_path + 'demographics/DEMO_H.XPT')
smoking_data = pd.read_sas(datasets_path + 'smoking/SMQ_H.XPT')
alcohol_data = pd.read_sas(datasets_path + 'alcohol_use/ALQ_H.XPT')
weight_data = pd.read_sas(datasets_path + 'weight_history/WHQ_H.XPT')
physical_data = pd.read_sas(datasets_path + 'physical_activity/PAQ_H.XPT')
pressurechol_data = pd.read_sas(datasets_path + 'blood_pressure/BPQ_H.XPT')
nutrition_data = pd.read_sas(datasets_path + 'diet_nutrition/DBQ_H.XPT')

#datasets keys or column fields
demographics_vars = list(demographics_data.keys())
smoking_vars =  list(smoking_data.keys())
alcohol_vars = list(alcohol_data.keys())
weight_vars = list(alcohol_data.keys())
physical_vars = list(physical_data.keys())
pressurechol_vars = list(pressurechol_data.keys())
nutrition_vars = list(nutrition_data.keys())

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

print("\nDiet and nutrition data exploration:\n")
print(nutrition_data[['SEQN','DBD895','DBD900', 'DBD905','DBD910']].head())
