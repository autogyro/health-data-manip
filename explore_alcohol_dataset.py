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
smoking_data = pd.read_sas(datasets_path + 'smoking/SMQ_H.XPT')
alcohol_data = pd.read_sas(datasets_path + 'alcohol_use/ALQ_H.XPT')
weight_data = pd.read_sas(datasets_path + 'weight_history/WHQ_H.XPT')
physical_data = pd.read_sas(datasets_path + 'physical_activity/PAQ_H.XPT')
pressurechol_data = pd.read_sas(datasets_path + 'blood_pressure/BPQ_H.XPT')
nutrition_data = pd.read_sas(datasets_path + 'diet_nutrition/DBQ_H.XPT')
consumer_data = pd.read_sas(datasets_path + 'consumer_behavior/CBQ_H.xpt')
insurance_data = pd.read_sas(datasets_path + 'health_insurance/HIQ_H.xpt')
healthcare_data = pd.read_sas(datasets_path + 'access_to_care/HUQ_H.xpt')
cardiovascular_data = pd.read_sas(datasets_path + 'cardiovascular/CDQ_H.XPT')
diabetes_data = pd.read_sas(datasets_path + 'diabetes/DIQ_H.XPT')


#clean_numdata(alcohol_data, 'ALQ130', lower_bound=0, upper_bound=25.0, headers_only=True)


alq101_data = alcohol_data.ALQ101
alq130_data = alcohol_data.ALQ130

print(alq101_data.head())




if False:

    print("\n********** --------- Acohol Dataset---------- *********\n")

    print("Alcohol consumption data description and stats:\n")
    print(alcohol_data.keys())


    #-----------------------------------------------------------------

    print("\n----ALQ101 variable-----:\n")
    print(alcohol_data.ALQ101.head(5))
    print(alcohol_data.ALQ101.describe())

    print("\nExcecisve numbers:\n")

    for index, row in alcohol_data.iterrows():

        if row['ALQ101'] > 2:
            print("Row index = {}".format(index))
            print(row['SEQN'], row['ALQ101'])

    print("\nNaN values:\n")

    cnan = 0
    for index, row in alcohol_data.iterrows():
        if pd.isnull(row['ALQ101']):
            cnan +=1

    print("Total number of NaN values in ALQ101 data: {}".format(cnan))

    # -----------------------------------------------------------------

    print("\n-----ALQ130 variable-----:\n")
    print(alcohol_data.ALQ130.head(5))
    print(alcohol_data.ALQ130.describe())

    print("\nExcecisve numbers:\n")

    for index, row in alcohol_data.iterrows():
        if row['ALQ130'] > 25:
            print(row['SEQN'], row['ALQ130'])

    print("\nNaN values:\n")

    cnan = 0
    for index, row in alcohol_data.iterrows():
        if pd.isnull(row['ALQ130']):
            cnan +=1

    print("Total number of NaN values in ALQ130 data: {}".format(cnan))





