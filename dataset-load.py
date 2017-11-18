#Testing loading datasets
#@Juan E. Rolon

import pandas as pd
import numpy as np

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




#datasets keys or column fields
demographics_vars = list(demographics_data.keys())
smoking_vars =  list(smoking_data.keys())
alcohol_vars = list(alcohol_data.keys())
weight_vars = list(alcohol_data.keys())
physical_vars = list(physical_data.keys())
pressurechol_vars = list(pressurechol_data.keys())
nutrition_vars = list(nutrition_data.keys())
consumer_vars = list(consumer_data.keys())
insurance_vars = list(insurance_data.keys())
healthcare_vars = list(healthcare_data.keys())
cardiovascular_vars = list(cardiovascular_data.keys())
diabetes_vars = list(diabetes_data.keys())


#datasets' quick preview of a subset of the column fields (variables of interest)

##### DEMOGRAPHICS ######

print("\nDemographics data exploration:\n")

#Print head of data on selected variables of interest
if False:
    print("Data head selected\n")
    print(demographics_data[['SEQN', 'RIDAGEYR', 'RIAGENDR', 'INDHHIN2']].head())
    print("")

seqn_num_records = demographics_data.SEQN.count()
age_num_records = demographics_data.RIDAGEYR.count()

max_age = np.max(demographics_data.RIDAGEYR)
min_age = np.min(demographics_data.RIDAGEYR)
min_age = np.min(demographics_data.RIDAGEYR)


print("SEQN num. records = {}, Age num records = {}".format(seqn_num_records, age_num_records))
print("Max age = {}".format(max_age))
print("Min age = {}".format(min_age))

print("\nDescriptive statistics of demographics data on selected variables:")
print(demographics_data[['SEQN', 'RIDAGEYR', 'INDHHIN2']].describe())

#---------------------------------- Age restricted data -----------------------------------

print("\nAge restricted data:\n")
age_gt18_data = demographics_data[(18.0 <= demographics_data.RIDAGEYR)]
age_lt45_data = demographics_data[(demographics_data.RIDAGEYR >= 45.0)]

#group of interest
age_gt18_lt45_data = age_gt18_data[(age_gt18_data.RIDAGEYR <=45.0)]
nag1845 = age_gt18_lt45_data.RIDAGEYR.count()

print("Num records of restricted age data (18 <= age <= 45: {}".format(nag1845))
print("\nDescriptive statistics of age-restricted data on selected variables:")
print(age_gt18_lt45_data[['SEQN', 'RIDAGEYR', 'INDHHIN2']].describe())

n_males = age_gt18_lt45_data[age_gt18_lt45_data.RIAGENDR == 1.0].RIAGENDR.count()
n_females = age_gt18_lt45_data[age_gt18_lt45_data.RIAGENDR == 2.0].RIAGENDR.count()
print("")
print("Num males = {}, Num females = {} on restricted age data".format(n_males, n_females))


#Random sample
print("Random sample (5 individuals) of age-restricted data:\n")
sample = age_gt18_lt45_data[['SEQN', 'RIDAGEYR', 'INDHHIN2']].sample(n=5, random_state=2131497)
print(sample)

#List containing all sequence numbers in age-restricted data
seqn_list = list(age_gt18_lt45_data.SEQN)

#---------------------------------- Alcohol use data -----------------------------------
print("\nAlcohol use data for given sample of sequence numbers:\n")

for record_num in list(sample.SEQN):
    print("SEQN = {}:".format(record_num))
    subrecord = alcohol_data[alcohol_data.SEQN == record_num][['SEQN','ALQ120Q', 'ALQ130']]
    print(subrecord, "\n")

ct=0
for record_num in list(age_gt18_lt45_data.SEQN):
    if not alcohol_data[alcohol_data.SEQN == record_num].empty:
        ct +=1
print("Number of non-empty datframes in Alcohol use data - for listed records: {}".format(ct))



#----------------------------------------------------------------------------------------
print("Age and alcohol data transformations:\n")

ageSeries = pd.Series(list(age_gt18_lt45_data.RIDAGEYR), index=list(age_gt18_lt45_data.SEQN))
tmp_alcoholSeries = pd.Series(list(alcohol_data.ALQ130), index=list(alcohol_data.SEQN))

print("\nAge series description:\n")
print(ageSeries.describe())
print("\nFull alcohol series description:\n")
print(tmp_alcoholSeries.describe())

print("\nReindexing of alcohol data series:\n")
print("Series head:\n")
alcohol_Series = tmp_alcoholSeries.reindex(list(age_gt18_lt45_data.SEQN))

print(tmp_alcoholSeries.describe())
print("\nAlcohol series restricted to age group stat description:\n")
print(alcohol_Series.describe())

alcohol_Series.to_csv("alcohol_series.csv", sep=',', header=True)
ageSeries.to_csv("age_series.csv", sep=',', header=True)







#Check dataframe head and compare with extracted series head
if False:
    print("Original age data frame head:\n")
    print(age_gt18_lt45_data[['SEQN', 'RIDAGEYR']].head())
    print("Extracted series age data head witn SEQN as index:\n")
    print(ageSeries.head())

    print("Original alcohol data frame head:\n")
    print(alcohol_data[['SEQN', 'ALQ130']].head())
    print("Extracted Series alcohol data head witn SEQN as index:\n")
    print(alcoholSeries.head())



#Extract a sample of individual seqn indexes from age-restricted data
#These indexes will help extract individual records from the remaining datasets
if False:
    ct = 0
    for seqn in age_gt18_lt45_data.SEQN:
        print(seqn)
        ct +=1
        if(ct >5):
            break


#Save age-restricted data ot CSV files
if False:
    age_gt18_lt45_data.to_csv('age_gt18_lt45.csv', sep=',')
    age_lt45_data.to_csv('age_lt45.csv', sep=',')
    age_gt18_data.to_csv('age_gt18.csv', sep=',')

#print age-restricted data heads
if False:
    print(age_gt18_data.head())
    print(age_lt45_data.head())
    print(age_gt18_lt45_data.head())



if False:

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

    print("\nConsumer behavior data exploration:\n")
    print(consumer_data[['SEQN','CBD070','CBD120', 'CBD130','CBD090']].head())

    print("\nHealth insurance data exploration:\n")
    print(insurance_data[['SEQN','HIQ011']].head())

    print("\nAccess to healthcare data exploration:\n")
    print(healthcare_data[['SEQN','HUQ010', 'HUQ020', 'HUQ051']].head())

    print("\nCardiovascular health data exploration:\n")
    print(cardiovascular_data[['SEQN','CDQ008', 'CDQ010', 'CDQ001']].head())

    print("\nDiabetes data exploration:\n")
    print(diabetes_data[['SEQN','DIQ010', 'DIQ160', 'DIQ170']].head())



