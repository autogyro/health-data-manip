# Script will load and process all datasets in selected NHANES 2013-2014 surveys
# @Juan E. Rolon
# https://github.com/juanerolon

import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import local modules
from importlib.machinery import SourceFileLoader
mod_path = "/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/nhanes2013-2104/"
txt = SourceFileLoader("text_utils", mod_path+"text_utils.py").load_module()
gut = SourceFileLoader("graph_utils", mod_path+"graph_utils.py").load_module()


#Path to datasets folders

datasets_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/'
project_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/'

#Process demographics dataframe
###############################
print("")
demographics_data = pd.read_sas(datasets_path + 'demographics/DEMO_H.XPT')
demographics_data = txt.switch_df_index(demographics_data, 'SEQN')
demographics_data = demographics_data[['RIDAGEYR', 'RIAGENDR','RIDRETH3','INDHHIN2']]
#Restrict dataframe to selected columns
demographics_data = txt.restrict_by_interval(demographics_data, 'RIDAGEYR', 18, 65, 'inclusive')
#Rename original columns to user specified names
demographics_data.rename(columns = {'RIDAGEYR':'AGE', 'RIAGENDR':'GENDER',
                            'RIDRETH3':'ETHNICITY','INDHHIN2':'INCOME_LEVEL'}, inplace=True)

demo_features = ['AGE','GENDER','ETHNICITY','INCOME_LEVEL']

demographics_data.INCOME_LEVEL.fillna(value=7, inplace=True)
for val in [77.0,99.0]:
    demographics_data['INCOME_LEVEL'].replace(to_replace=val, value=7, inplace=True)
#Downcast selected features
for feat in demo_features:
    demographics_data[feat] = pd.to_numeric(demographics_data[feat], downcast='integer')


# txt.headcounts(demographics_data)

#Process alcohol consumption dataframe
######################################
print("")
alcohol_data = pd.read_sas(datasets_path + 'alcohol_use/ALQ_H.XPT')
alcohol_data = txt.switch_df_index(alcohol_data, 'SEQN')
alcohol_data = alcohol_data[['ALQ130']]
alcohol_data.rename(columns = {'ALQ130':'ALCOHOL_NUM'}, inplace=True)
alcohol_data['ALCOHOL_NUM'].fillna(value=0, inplace=True)
alcohol_data['ALCOHOL_NUM'] = pd.to_numeric(alcohol_data['ALCOHOL_NUM'], downcast='integer')
alcohol_data.ALCOHOL_NUM.replace(to_replace=999, value=0, inplace=True)

alcohol_features = ['ALCOHOL_NUM']

# txt.headcounts(alcohol_data)

#Process smoking consumption dataframe
######################################
print("")
smoking_data = pd.read_sas(datasets_path + 'smoking/SMQ_H.XPT')
smoking_data = txt.switch_df_index(smoking_data, 'SEQN')
smoking_data = smoking_data[['SMQ040']]
smoking_data.rename(columns = {'SMQ040':'SMOKING'}, inplace=True)
smoking_data['SMOKING'].fillna(value=0, inplace=True)
smoking_data['SMOKING'] = pd.to_numeric(smoking_data['SMOKING'], downcast='integer')
smoking_data.SMOKING.replace(to_replace=1, value=1, inplace=True)
smoking_data.SMOKING.replace(to_replace=2, value=1, inplace=True)
smoking_data.SMOKING.replace(to_replace=0, value=0, inplace=True)
smoking_data.SMOKING.replace(to_replace=3, value=0, inplace=True)

smoking_features = ['SMOKING']

# txt.headcounts(smoking_data)

#Process weight data
####################
print("")
weight_data = pd.read_sas(datasets_path + 'weight_history/WHQ_H.XPT')
weight_data = txt.switch_df_index(weight_data, 'SEQN')
weight_data = weight_data[['WHD010', 'WHD020']]
weight_data.dropna(axis=0, how='any', inplace=True)

weight_data = weight_data[weight_data.WHD010 != 7777.0]
weight_data = weight_data[weight_data.WHD010 != 9999.0]
weight_data = weight_data[weight_data.WHD020 != 7777.0]
weight_data = weight_data[weight_data.WHD020 != 9999.0]

def bmi(h, w):
    return np.round((w/h**2) * 703.0, 2)

weight_data['BMI'] = bmi(weight_data['WHD010'], weight_data['WHD020'])
weight_data.drop(['WHD010', 'WHD020'], axis=1, inplace=True)

weight_features = ['BMI']

# txt.headcounts(weight_data)


#Process nutrition data
#######################
nutrition_data = pd.read_sas(datasets_path + 'diet_nutrition/DBQ_H.XPT')
nutrition_data = txt.switch_df_index(nutrition_data, 'SEQN')
nutrition_data = nutrition_data[['DBD895', 'DBD900']]
nutrition_data.rename(columns = {'DBD895':'NOTHOME_FOOD', 'DBD900':'FAST_FOOD' }, inplace=True)
nutrition_data .dropna(axis=0, how='any', inplace=True)
nutrition_data['NOTHOME_FOOD'] = pd.to_numeric(nutrition_data['NOTHOME_FOOD'], downcast='integer')
nutrition_data['FAST_FOOD'] = pd.to_numeric(nutrition_data['FAST_FOOD'], downcast='integer')
#-----data imputation
mean_val1 = int(np.round(nutrition_data['NOTHOME_FOOD'].values.mean()))
nutrition_data.NOTHOME_FOOD.replace(to_replace=5555, value=mean_val1, inplace=True)
nutrition_data.NOTHOME_FOOD.replace(to_replace=9999, value=mean_val1, inplace=True)
mean_val2 = int(np.round(nutrition_data['FAST_FOOD'].values.mean()))
nutrition_data.FAST_FOOD.replace(to_replace=5555, value=mean_val2, inplace=True)
nutrition_data.FAST_FOOD.replace(to_replace=9999, value=mean_val2, inplace=True)

#-----data transformation to percentage values
def percent_transform(x,n):
    return np.round(float(x/n *100.0),2)

nutrition_data['NOTHOME_FOOD'] = nutrition_data['NOTHOME_FOOD'].apply(lambda x: percent_transform(x, 21))
nutrition_data['FAST_FOOD'] = nutrition_data['FAST_FOOD'].apply(lambda x: percent_transform(x, 21))

nutrition_features = ['NOTHOME_FOOD', 'FAST_FOOD']

# txt.headcounts(nutrition_data)


#Process nutrition data
#######################
insurance_data = pd.read_sas(datasets_path + 'health_insurance/HIQ_H.xpt')
insurance_data = txt.switch_df_index(insurance_data, 'SEQN')
insurance_data = insurance_data[['HIQ011']]
insurance_data.rename(columns = {'HIQ011':'INSURANCE'}, inplace=True)
insurance_data['INSURANCE'].fillna(value=0, inplace=True)
insurance_data['INSURANCE'] = pd.to_numeric(insurance_data['INSURANCE'], downcast='integer')
for val in [2,7,9]:
    insurance_data.INSURANCE.replace(to_replace=val, value=0, inplace=True)

insurance_features = ['INSURANCE']

# txt.headcounts(insurance_data)
# txt.get_feature_counts(insurance_data, ['INSURANCE'])
# txt.count_feature_nans(insurance_data, ['INSURANCE'])


#Process blood-pressure cholesterol text data
#############################################

cholpressure_data = pd.read_sas(datasets_path + 'blood_pressure/BPQ_H.XPT')
cholpressure_data = txt.switch_df_index(cholpressure_data, 'SEQN')
cholpressure_data = cholpressure_data[['BPQ020','BPQ030', 'BPQ040A', 'BPQ050A', 'BPQ080', 'BPQ090D']]

old_names = ['BPQ020','BPQ030', 'BPQ040A', 'BPQ050A', 'BPQ080', 'BPQ090D']
cholpressure_features = ['HYPERTENSION_ONSET', 'HYPERTENSION_1', 'HYPERTENSION_2', 'HYPERTENSION_3', 'HIGHCHOL_ONSET', 'HIGHCHOL']

for n, new_name in enumerate(cholpressure_features):
    cholpressure_data.rename(columns={old_names[n]: new_name}, inplace=True)

#------global nans imputation
cholpressure_data.fillna(value=0, inplace=True)

for feat in cholpressure_features:
    cholpressure_data[feat] = pd.to_numeric(cholpressure_data[feat], downcast='integer')
#-----missing values imputation
for val in [2,7,9]:
    cholpressure_data.replace(to_replace=val, value=0, inplace=True)
#-----combine HYPERTENSION features
cholpressure_data['HYPERTENSION'] = np.vectorize(txt.bit_logic)\
    (cholpressure_data['HYPERTENSION_1'].values, cholpressure_data['HYPERTENSION_2'].values, 'OR')

cholpressure_data['HYPERTENSION'] = np.vectorize(txt.bit_logic)\
    (cholpressure_data['HYPERTENSION'].values, cholpressure_data['HYPERTENSION_3'].values, 'OR')

cholpressure_data.drop(['HYPERTENSION_1', 'HYPERTENSION_2', 'HYPERTENSION_3'], axis=1, inplace=True)

cholpressure_features = ['HYPERTENSION_ONSET', 'HYPERTENSION', 'HIGHCHOL_ONSET', 'HIGHCHOL']

# txt.headcounts(cholpressure_data)
# txt.get_feature_counts(cholpressure_data, cholpressure_features)


#Process cardiovascular text data
#############################################
cardiovascular_data = pd.read_sas(datasets_path + 'cardiovascular/CDQ_H.XPT')  #DONE
cardiovascular_data = txt.switch_df_index(cardiovascular_data, 'SEQN')
cardiovascular_data = cardiovascular_data[['CDQ001', 'CDQ008', 'CDQ010']]
cardiovascular_data.rename(columns = {'CDQ001':'CHEST_DISCOMFORT'}, inplace=True)
cardiovascular_data.rename(columns = {'CDQ008':'CHEST_PAIN_30MIN'}, inplace=True)
cardiovascular_data.rename(columns = {'CDQ010':'BREATH_SHORTNESS'}, inplace=True)

cardio_features = ['CHEST_DISCOMFORT', 'CHEST_PAIN_30MIN', 'BREATH_SHORTNESS']

cardiovascular_data.fillna(value=0, inplace=True)
for feat in cardio_features:
    cardiovascular_data[feat] = pd.to_numeric(cardiovascular_data[feat], downcast='integer')
for val in [2,7,9]:
    cardiovascular_data.replace(to_replace=val, value=0, inplace=True)


# txt.headcounts(cardiovascular_data)



#Process diabetes text data
#############################################
diabetes_data = pd.read_sas(datasets_path + 'diabetes/DIQ_H.XPT') #DONE
diabetes_data = txt.switch_df_index(diabetes_data, 'SEQN')
diabetes_data = diabetes_data[['DIQ175A','DIQ010', 'DIQ160', 'DIQ170']]
old_diab_features = ['DIQ175A','DIQ010', 'DIQ160', 'DIQ170']

diabetes_features = ['FAMILIAL_DIABETES', 'DIAGNOSED_DIABETES', 'DIAGNOSED_PREDIABETES', 'RISK_DIABETES']

for n, new_name in enumerate(diabetes_features):
    diabetes_data.rename(columns={old_diab_features[n]: new_name}, inplace=True)
diabetes_data.fillna(value=0, inplace=True)
for feat in diabetes_features:
    diabetes_data[feat] = pd.to_numeric(diabetes_data[feat], downcast='integer')

diabetes_data['FAMILIAL_DIABETES'].replace(to_replace=10, value=1, inplace=True)
for val in [77,99]:
    diabetes_data['FAMILIAL_DIABETES'].replace(to_replace=val, value=0, inplace=True)

for val in [1,3]:
    diabetes_data['DIAGNOSED_DIABETES'].replace(to_replace=val, value=1, inplace=True)
for val in [2,7,9]:
    diabetes_data['DIAGNOSED_DIABETES'].replace(to_replace=val, value=0, inplace=True)

for val in [2,7,9]:
    diabetes_data['DIAGNOSED_PREDIABETES'].replace(to_replace=val, value=0, inplace=True)
    diabetes_data['RISK_DIABETES'].replace(to_replace=val, value=0, inplace=True)

# txt.headcounts(diabetes_data)


#REINDEX DATAFRAMES TO CONFORM TO DEMOGRAPHICS DATA SEQN INDEX
##############################################################

#Process alcohol consumption dataframe
#-------------------------------------
print("\nReindex alcohol dataframe using indexes from demographics dataframe")
txt.bitwise_index_compare(demographics_data,alcohol_data) # compare indexes respect to master dataframe (demographics)
alcohol_data = alcohol_data.reindex(demographics_data.index) # reindex
mean_alcohol = int(np.round(alcohol_data.mean())) #compute mean values of features
alcohol_data.fillna(value=mean_alcohol, inplace=True) # impute mean values replacing nans

# txt.count_feature_nans(alcohol_data, alcohol_features)
# txt.headcounts(alcohol_data)

# print(alcohol_data.loc[index_set_1])
# print(demographics_data.AGE.loc[index_set_1])

#Process smoking consumption dataframe
#-------------------------------------
print("\nReindex smoking dataframe using indexes from alcohol dataframe")
txt.bitwise_index_compare(alcohol_data, smoking_data)
smoking_data = smoking_data.reindex(alcohol_data.index)
# txt.count_feature_nans(smoking_data, smoking_features)
# txt.headcounts(smoking_data)

#Process weight history dataframe
#-------------------------------------
print("\nReindex weight dataframe using indexes from smoking dataframe")
txt.bitwise_index_compare(smoking_data,weight_data)
weight_data = weight_data.reindex(smoking_data.index)
txt.count_feature_nans(weight_data,weight_features)
mean_bmi = int(np.round(weight_data['BMI'].mean())) #compute mean values of features
weight_data.fillna(value=mean_bmi, inplace=True)
# txt.count_feature_nans(weight_data,weight_features)
# txt.headcounts(weight_data,2)


#Process nutrition dataframe
#-------------------------------------
print("\nReindex nutrition dataframe using indexes from weight history dataframe")
txt.bitwise_index_compare(weight_data,nutrition_data)

nutrition_data = nutrition_data.reindex(weight_data.index)

#txt.count_feature_nans(nutrition_data, nutrition_features)

#txt.headcounts(nutrition_data)

for feat in nutrition_features:
    mval = np.round(nutrition_data[feat].mean(),2)
    nutrition_data[feat].fillna(value=mval,inplace=True)
    print(mval)

#txt.headcounts(nutrition_data)
#txt.count_feature_nans(nutrition_data, nutrition_features)

#Process blood-pressure cholesterol dataframe
#--------------------------------------------

print("\nReindex cholesterol blood-pressure dataframe using indexes from nutrition dataframe")
txt.bitwise_index_compare(nutrition_data,cholpressure_data)
cholpressure_data = cholpressure_data.reindex(nutrition_data.index)
#txt.headcounts(cholpressure_data)
#txt.count_feature_nans(cholpressure_data, cholpressure_features)

#Process cardiovascular dataframe
#--------------------------------------------
print("\nReindex cardiovascular dataframe using indexes from cholpressure dataframe")
txt.bitwise_index_compare(cholpressure_data,cardiovascular_data)
cardiovascular_data = cardiovascular_data.reindex(cholpressure_data.index)

#txt.headcounts(cardiovascular_data)
#txt.count_feature_nans(cardiovascular_data, cardio_features)

for feat in cardio_features:
    val = 0
    cardiovascular_data[feat].fillna(value=val,inplace=True)

#txt.headcounts(cardiovascular_data)
#txt.count_feature_nans(cardiovascular_data, cardio_features)

#Process diabetes dataframe
#--------------------------------------------
print("\nReindex diabetes dataframe using indexes from cardiovascular dataframe")
txt.bitwise_index_compare(cardiovascular_data, diabetes_data)
diabetes_data = diabetes_data.reindex(demographics_data.index)
#txt.count_feature_nans(diabetes_data, diabetes_features)


#CONCATENATE ALL DATAFRAMES
##############################################################

nhanes1314_datasets_p1 = [demographics_data, alcohol_data, smoking_data, weight_data, nutrition_data, cholpressure_data,
                         cardiovascular_data, diabetes_data]

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
nhanes_2013_2014_df1 = pd.concat(nhanes1314_datasets_p1, axis=1)


#*************************************
#Touch up pre-processing

#GENDER CATEGORICAL
nhanes_2013_2014_df1.GENDER.replace(to_replace=1, value='Male', inplace=True)
nhanes_2013_2014_df1.GENDER.replace(to_replace=2, value='Female', inplace=True)

#ETHNICITY CATEGORICAL
nhanes_2013_2014_df1.ETHNICITY.replace(to_replace=1, value='Hispanic', inplace=True)
nhanes_2013_2014_df1.ETHNICITY.replace(to_replace=2, value='Hispanic', inplace=True)
nhanes_2013_2014_df1.ETHNICITY.replace(to_replace=3, value='White', inplace=True)
nhanes_2013_2014_df1.ETHNICITY.replace(to_replace=4, value='Black', inplace=True)
nhanes_2013_2014_df1.ETHNICITY.replace(to_replace=6, value='Asian', inplace=True)
nhanes_2013_2014_df1.ETHNICITY.replace(to_replace=7, value='Other', inplace=True)

#INCOME_LEVEL (Make income level to be in monotonic increasing order)
nhanes_2013_2014_df1.INCOME_LEVEL.replace(to_replace=12, value=6, inplace=True)
nhanes_2013_2014_df1.INCOME_LEVEL.replace(to_replace=13, value=3, inplace=True)
nhanes_2013_2014_df1.INCOME_LEVEL.replace(to_replace=14, value=11, inplace=True)
nhanes_2013_2014_df1.INCOME_LEVEL.replace(to_replace=15, value=12, inplace=True)
#after this change, income level = 12 becomes max level


#ONE-HOT ENCODING

nhanes_2013_2014_df1 = pd.get_dummies(nhanes_2013_2014_df1, columns=['GENDER', 'ETHNICITY'])

#MIN-MAX SCALING

from sklearn.preprocessing import MinMaxScaler
scaled_features = ['AGE', 'INCOME_LEVEL', 'ALCOHOL_NUM', 'BMI', 'NOTHOME_FOOD', 'FAST_FOOD']
scaler = MinMaxScaler()  # default=(0, 1)
nhanes_2013_2014_df1[scaled_features] = scaler.fit_transform(nhanes_2013_2014_df1[scaled_features])

print(nhanes_2013_2014_df1.head())
#txt.count_feature_nans(nhanes_2013_2014_df1, list(nhanes_2013_2014_df1.columns))


csv_filename_qdata = 'questionnaire_data.csv'
nhanes_2013_2014_df1.to_csv(csv_filename_qdata)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

txt.headcounts(nhanes_2013_2014_df1)
txt.count_feature_nans(nhanes_2013_2014_df1, list(nhanes_2013_2014_df1.columns))


#PROCESS STANDARD BIOCHEMISTRY DATASET
##############################################################
biochemistry_data = pd.read_sas(datasets_path + '__standard_biochem/BIOPRO_H.XPT')

#CODES:
#........ LBXSAL - Albumin (g/dL)
#........ LBDSALSI - Albumin (g/L)   ** D
#........ LBXSAPSI - Alkaline phosphatase (IU/L)

#........ LBXSASSI - Aspartate aminotransferase AST (IU/L)
#........ LBXSATSI - Alanine aminotransferase ALT (IU/L)

#........ LBXSBU - Blood urea nitrogen (mg/dL)
#........ LBDSBUSI - Blood urea nitrogen (mmol/L)  ** D
#........ LBXSC3SI - Bicarbonate (mmol/L)

#........ LBXSCA - Total calcium (mg/dL)

#........ LBDSCASI - Total calcium (mmol/L)  ** D
#........ LBXSCH - Cholesterol (mg/dL)
#........ LBDSCHSI - Cholesterol (mmol/L)  ** D

#........ LBXSCK - Creatine Phosphokinase(CPK) (IU/L)


#........ LBXSCLSI - Chloride (mmol/L)
#........ LBXSCR - Creatinine (mg/dL)
#........ LBDSCRSI - Creatinine (umol/L)  ** D

#........ LBXSGB - Globulin (g/dL)

#........ LBDSGBSI - Globulin (g/L)   ** D
#........ LBXSGL - Glucose, refrigerated serum (mg/dL)
#........ LBDSGLSI - Glucose, refrigerated serum (mmol/L)  ** D
#........ LBXSGTSI - Gamma glutamyl transferase (U/L)
#........ LBXSIR - Iron, refrigerated serum (ug/dL)
#........ LBDSIRSI - Iron, refrigerated serum (umol/L)  ** D
#........ LBXSKSI - Potassium (mmol/L)
#........ LBXSLDSI - Lactate dehydrogenase (U/L)

#........ LBXSNASI - Sodium (mmol/L)
#........ LBXSOSSI - Osmolality (mmol/Kg)

#........ LBXSPH - Phosphorus (mg/dL)
#........ LBDSPHSI - Phosphorus (mmol/L)   ** D
#........ LBXSTB - Total bilirubin (mg/dL)
#........ LBDSTBSI - Total bilirubin (umol/L)     ** D

#........ LBXSTP - Total protein (g/dL)

#........ LBDSTPSI - Total protein (g/L)  ** D
#........ LBXSTR - Triglycerides, refrigerated (mg/dL)
#........ LBDSTRSI - Triglycerides, refrigerated (mmol/L) ** D
#........ LBXSUA - Uric acid (mg/dL)
#........ LBDSUASI - Uric acid (umol/L) ** D
#

#discard redundant features
bio_redundant_features = ['LBDSUASI', 'LBDSTRSI', 'LBDSTPSI', 'LBDSTBSI', 'LBDSPHSI', 'LBDSIRSI', 'LBDSGLSI', 'LBDSGBSI',
                          'LBDSCRSI', 'LBDSCHSI', 'LBDSCASI', 'LBDSBUSI', 'LBDSALSI']

biochemistry_data.drop(bio_redundant_features, axis=1, inplace=True)
biochemistry_data = txt.switch_df_index(biochemistry_data, 'SEQN')

biochem_features = list(biochemistry_data.columns)

#Intial removal of records with missing values
biochemistry_data.dropna(axis=0, how='any', inplace=True) #Remove all records with missing biochemistry data

txt.bitwise_index_compare(nhanes_2013_2014_df1,biochemistry_data)
biochemistry_data = biochemistry_data.reindex(nhanes_2013_2014_df1.index)

#Remove again records with missing alues after reindexing
biochemistry_data.dropna(axis=0, how='any', inplace=True) #Remove all records with missing biochemistry data


if False:
    #txt.count_feature_nans(biochemistry_data,biochem_features)
    txt.count_rows_with_nans(biochemistry_data)
    lsn = txt.get_nanrows_indexes(biochemistry_data)
    print(biochemistry_data.head())


if False:
    #-------------------- MERGE QUESTIONAIRE AND BIOCHEMISTRY DATASETS -------------------------
    ############################################################################################
    questionaire_data = nhanes_2013_2014_df1.copy(deep=True)

    questionaire_data = questionaire_data.reindex(biochemistry_data.index)
    txt.count_rows_with_nans(questionaire_data)

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    nhanes_2013_2014_full_data = pd.concat([biochemistry_data, questionaire_data],axis=1)
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


    #Check integrity of full data
    if False:
        features = list(nhanes_2013_2014_full_data.columns)
        txt.headcounts(nhanes_2013_2014_full_data)
        txt.count_feature_nans(nhanes_2013_2014_full_data, features)



if False:
    old_bio_features = ['LBXSAL', 'LBXSAPSI', 'LBXSASSI', 'LBXSATSI',
                        'LBXSBU', 'LBXSC3SI', 'LBXSCA', 'LBXSCH',
                        'LBXSCK', 'LBXSCLSI', 'LBXSCR', 'LBXSGB',
                        'LBXSGL', 'LBXSGTSI', 'LBXSIR', 'LBXSKSI',
                        'LBXSLDSI', 'LBXSNASI', 'LBXSOSSI', 'LBXSPH',
                        'LBXSTB', 'LBXSTP', 'LBXSTR', 'LBXSUA']

    new_bio_features = ['ALBUMIN(g/dL)', 'ALKALINE_PHOSPHATASE(IU/L)', 'ASPARTATE_AMINOTRANSFERASE_AST(IU/L)', 'ALANINE_AMINOTRANSFERASE_ALT(IU/L)',
                        'BLOOD_UREA_NITROGEN(mg/dL)', 'BICARBONATE(mmol/L)', 'TOTAL_CALCIUM(mg/dL)', 'CHOLESTEROL(mg/dL)',
                        'CREATINE_PHOSPHOKINASE(IU/L)', 'CHLORIDE(mmol/L)', 'CREATININE(mg/dL)', 'GLOBULIN(g/dL)',
                        'GLUCOSE_RSERUM(mg/dL)', 'GAMMA_GLUT_TRNSF(U/L)', 'IRON_RSERUM(ug/dL)', 'POTASSIUM(mmol/L)',
                        'LACTATE_DEHYDROGENASE(U/L)', 'SODIUM(mmol/L)', 'OSMOLALITY(mmol/Kg)', 'PHOSPHORUS(mg/dL)',
                        'TOTAL_BILIRUBIN(mg/dL)', 'TOTAL_PROTEIN(g/dL)', 'TRIGLYCERIDES(mg/dL)', 'URIC_ACID(mg/dL)']


if False:
    for n, new_name in enumerate(new_bio_features):
        biochemistry_data.rename(columns={old_bio_features[n]: new_name}, inplace=True)

if False:

    for n, new_name in enumerate(new_bio_features):
        nhanes_2013_2014_full_data.rename(columns={old_bio_features[n]: new_name}, inplace=True)


if False:
    from scipy.stats import normaltest
    cols = list(biochemistry_data.columns)
    for feat in cols:
        statistic, p = normaltest(biochemistry_data[feat].values)
        if p >= 0.05:
            print ("The feature {} likely has a normal distribution".format(feat))
        else:
            print ("The feature {} likely doesn't have a normal distribution".format(feat))


if False:
    txt.count_rows_with_nans(nhanes_2013_2014_full_data)
    txt.headcounts(nhanes_2013_2014_full_data)



#Test exported files
if False:
    nhanes_full_test = feather.read_dataframe(project_path + 'nhanes_2013_2014_full_data.feather')
    nhanes_full_test = nhanes_full_test.set_index('SEQN') #Restore the index saved during the export step



#First set of low-variance-carrying features
if False:

    low_var_features = ['LBXSAL', 'LBXSC3SI', 'LBXSCA', 'LBXSCH', 'LBXSCLSI', 'LBXSGB', 'LBXSGL', 'LBXSKSI',
                        'LBXSLDSI', 'LBXSNASI', 'LBXSOSSI', 'LBXSPH', 'LBXSTP']

    biochemistry_data.drop(low_var_features, axis=1, inplace=True)

    txt.headcounts(biochemistry_data)


#_____________________________________________ ANALYZE THIS STEP FURTHER _______________________
#Turn to false to include all low_var features
if True:

    low_var_features = ['LBXSAL', 'LBXSC3SI', 'LBXSCA', 'LBXSCH', 'LBXSCLSI', 'LBXSGB', 'LBXSGL', 'LBXSKSI',
                        'LBXSLDSI', 'LBXSNASI', 'LBXSOSSI', 'LBXSPH', 'LBXSTP']

    biochemistry_data.drop(low_var_features, axis=1, inplace=True)

#Turn to True to include all colinear features
if True:
    colinear_features = ['LBXSTB', 'LBXSASSI', 'LBXSCR', 'LBXSUA', 'LBXSAPSI']
    biochemistry_data.drop(colinear_features, axis=1, inplace=True)

    #txt.headcounts(biochemistry_data)

if True:
    #Transform biochemistry features

    # Log(x+1) transform
    features = list(biochemistry_data.columns)
    biochemistry_data[features] = biochemistry_data[features ].apply(lambda x: np.log(x + 1))

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()  # default=(0, 1)
    biochemistry_data[features] = scaler.fit_transform(biochemistry_data[features])

    #txt.headcounts(biochemistry_data)
    #txt.count_feature_nans(biochemistry_data, features)


if True:
    #-------------------- MERGE QUESTIONAIRE AND BIOCHEMISTRY DATASETS -------------------------
    ############################################################################################
    questionaire_data = nhanes_2013_2014_df1.copy(deep=True)

    questionaire_data = questionaire_data.reindex(biochemistry_data.index)

    if False:
        txt.count_rows_with_nans(questionaire_data)

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    nhanes_2013_2014_full_data = pd.concat([biochemistry_data, questionaire_data],axis=1)
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


    #Check integrity of full data
    if False:
        features = list(nhanes_2013_2014_full_data.columns)
        txt.headcounts(nhanes_2013_2014_full_data)
        txt.count_feature_nans(nhanes_2013_2014_full_data, features)

#EXPORT DATASETS

if True:
    biochemistry_data.to_csv('biochemistry_data.csv')


if False:
    import feather

    #Save Final stage merged dataframe
    filename_final = 'nhanes_2013_2014_full_data.feather'
    feather.write_dataframe(nhanes_2013_2014_full_data, filename_final)

    csv_filename = 'nhanes_2013_2014_full_data.csv'
    nhanes_2013_2014_full_data.to_csv(csv_filename)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initial models testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("\n............... Modeling....................\n")

from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score



if True:
    #Model input features
    model_features = pd.concat([biochemistry_data, questionaire_data.BMI],axis=1)
    model_features = pd.concat([model_features, questionaire_data.AGE], axis=1)
    model_features = pd.concat([model_features, questionaire_data.SMOKING], axis=1)

    #Target for prediction/classification

    #model_targets = questionaire_data.DIAGNOSED_PREDIABETES
    model_targets = questionaire_data.HIGHCHOL

    if False:
        features = list(model_features.columns)
        txt.headcounts(model_features)
        txt.count_feature_nans(model_features, features)


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(model_features,model_targets,test_size=0.35,random_state=0)

    #Classifiers


    from sklearn import tree
    from sklearn.metrics import fbeta_score
    from sklearn.metrics import accuracy_score

    clf = tree.DecisionTreeClassifier(random_state=10, max_depth=32, max_features=None)
    clf = clf.fit(X_train, y_train)
    predictions_train = clf.predict(X_train)
    predictions_test = clf.predict(X_test)

    acc_train = accuracy_score(y_train, predictions_train)
    acc_test = accuracy_score(y_test, predictions_test)

    beta = 0.5

    f_train = fbeta_score(y_train, predictions_train, beta=beta)
    f_test = fbeta_score(y_test, predictions_test, beta=beta)

    print("acc_train = {}, acc_test ={}".format(acc_train, acc_test))
    print("f_train = {}, f_test ={}".format(f_train, f_test))

sys.exit()


############################################# Initial Visual Tests #####################################################
########## ScatterMatrixPlot ##########


if False:
    #Transformed features

    pd.scatter_matrix(biochemistry_data, alpha = 0.3, figsize = (16,8), diagonal = 'kde')
    plt.show()

if False:

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    from sklearn.decomposition import PCA

    ndims = 2
    dim_labels = []
    for i in range(1, ndims + 1):
        dim_labels.append("Dimension {}".format(i))

    pca = PCA(n_components=ndims) #.....................No. of dimensions
    pca.fit(biochemistry_data)
    reduced_data = pca.transform(biochemistry_data)
    reduced_data = pd.DataFrame(reduced_data, columns = dim_labels)

    "K-Means Silhouette Scoring Tests:\n"
    for kn in range(2, 9):

        clm = KMeans(n_clusters=kn, random_state=0)
        clm.fit(reduced_data)
        preds = clm.predict(reduced_data)
        centers = clm.cluster_centers_
        #sample_preds = clm.predict(pca_samples)
        score = silhouette_score(reduced_data, preds, random_state=10)
        print("Number of clusters = {}, Score = {}".format(kn, np.round(score, 4)))

if False:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca.fit(biochemistry_data)
    reduced_data = pca.transform(biochemistry_data)
    reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
    gut.biplot(biochemistry_data, reduced_data, pca)
    plt.show()



if False:
    from sklearn.decomposition import PCA

    data = biochemistry_data
    pca = PCA(n_components=data.shape[1])
    pca.fit(data)
    pca_results = gut.pca_results(data, pca)

    print("Cumulative sum of of explained variance by dimension:")
    print(pca_results['Explained Variance'].cumsum())
    print("")
    print("PCA detailed results:")
    print(pca_results)

    csv_filename = 'pca_analysis_reduced_biodata2.csv'
    pca_results.to_csv(csv_filename)

#Generate pair of heatmaps for dimensios vs features corr and variance
#Suitable for reduced data pca analysis (above)
if False:
    import seaborn as sns

    cols = list(pca_results.columns)[1:]
    rows = list(pca_results.index)
    pca_matrix = pca.components_
    pca_squared_matrix = np.square(pca.components_)

    plt.figure(1, figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.set(font_scale=1.0)
    heat_map = sns.heatmap(pca_matrix, cbar=True, annot=True, square=True, fmt='.4f',
                           annot_kws={'size': 8}, yticklabels=rows, xticklabels=cols)
    plt.title("Feature loadings correlation matrix")
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')

    plt.subplot(1, 2, 2)
    sns.set(font_scale=1.0)
    heat_map = sns.heatmap(pca_squared_matrix, cmap="YlGnBu", cbar=True, annot=True, square=True, fmt='.4f',
                           annot_kws={'size': 8}, yticklabels=rows, xticklabels=cols)
    plt.title("Variance percentages explained by feature")
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')

    plt.tight_layout()
    plt.show()

#PCA of full biochem data
if False:
    from sklearn.decomposition import PCA

    data = biochemistry_data

    pca = PCA(n_components=data.shape[1])

    pca_results = gut.pca_results(data, pca)

    print("Cumulative sum of of explained variance by dimension:")
    print(pca_results['Explained Variance'].cumsum())
    print("")
    print("PCA detailed results:")
    print(pca_results)

    csv_filename = 'pca_analysis_biochemistry.csv'
    pca_results.to_csv(csv_filename)



#Bar plot of pca results
if False:
    from sklearn.decomposition import PCA
    data = biochemistry_data


    pca = PCA(n_components=data.shape[1])
    pca.fit(data)
    pca_results = gut.pca_results(data, pca)
    plt.show()


########## Box Plots ##########
if False:

    feats = []
    for m in range(12, 24):
        feats.append(old_bio_features[m])

    data = biochemistry_data[feats]


    #sns.reset_orig()
    plt.figure(1, figsize=(16, 7))

    plt.subplot(1, 1, 1)
    plt.title('Box plot scaled data - includes all outliers')
    data.boxplot(showfliers=True)
    plt.ylim(0,15)
    plt.show()



########## Heatmap ##########
if False:
    import seaborn as sns

    data = biochemistry_data

    #old code block
    if False:

        feats = []
        for m in range(0,24):
            feats.append(old_bio_features[m])

        data = biochemistry_data[feats]
        print(data.head())

    cols = list(data.columns)
    corr_matrix = np.corrcoef(data[cols].values.T)

    plt.figure(1, figsize=(10, 8))
    sns.set(font_scale=1.0)
    heat_map = sns.heatmap(corr_matrix, cbar=True, annot=True, square=True, fmt='.2f',
                           annot_kws={'size': 12}, yticklabels=cols, xticklabels=cols)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.tight_layout()
    plt.show()



