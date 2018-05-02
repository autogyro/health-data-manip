

import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from importlib.machinery import SourceFileLoader
from itertools import combinations

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import time



#Specify whether working on local or remote instance
local = 1

if (local ==1):
    mod_path = "/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/nhanes2013-2104/"
    txt = SourceFileLoader("text_utils", mod_path+"text_utils.py").load_module()
    gut = SourceFileLoader("graph_utils", mod_path+"graph_utils.py").load_module()


    #Path to datasets folders
    datasets_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/'
    project_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/'
elif (local == 2):
    mod_path = "D:\\Dropbox\\Dropbox\\_machine_learning\\udacity_projects\\capstone\\gits\\nhanes2013-2104\\"
    txt = SourceFileLoader("text_utils", mod_path+"text_utils.py").load_module()
    gut = SourceFileLoader("graph_utils", mod_path+"graph_utils.py").load_module()

    #Path to datasets folders
    datasets_path = "D:\\Dropbox\\Dropbox\\_machine_learning\\udacity_projects\\capstone\\gits\\health-data-manip\\datasets\\"
    project_path = "D:\\Dropbox\\Dropbox\\_machine_learning\\udacity_projects\\capstone\\gits\\health-data-manip\\"

elif (local == 3):

    mod_path = "/home/aind2/capstone/gits/NHANES-2013-2014/"
    txt = SourceFileLoader("text_utils", mod_path + "text_utils.py").load_module()
    gut = SourceFileLoader("graph_utils", mod_path + "graph_utils.py").load_module()

    # Path to datasets folders

    datasets_path = '/home/aind2/capstone/gits/health-data-manip/datasets/'
    project_path = '/home/aind2/capstone/gits/health-data-manip/'

else:
    raise Exception("Invalid local folder selection")



#------------------------------------------------------------------------------------------------------------------------

full_data = pd.read_csv(project_path + 'nhanes_2013_2014_full_data.csv',index_col=0)


full_data.drop(['ETHNICITY_Other', 'GENDER_Male', 'GENDER_Female','DIAGNOSED_PREDIABETES',
                'BREATH_SHORTNESS', 'CHEST_PAIN_30MIN', 'CHEST_DISCOMFORT', 'RISK_DIABETES',
                'FAST_FOOD', 'NOTHOME_FOOD', 'INCOME_LEVEL','ETHNICITY_White', 'ETHNICITY_Black',
                'ETHNICITY_Hispanic', 'ETHNICITY_Asian', 'SMOKING'], axis = 1, inplace=True)

#Select the biofeatures to drop; the less you drop the more combinations will be calculated
if False:
    full_data.drop(['LBXSTP', 'LBXSPH', 'LBXSC3SI', 'LBXSCA', 'LBXSLDSI', 'LBXSCK',
                    'LBXSCH', 'LBXSUA', 'LBXSASSI', 'LBXSIR'], axis = 1, inplace=True)

if True:
    full_data.drop(['LBXSTP', 'LBXSPH', 'LBXSC3SI', 'LBXSCA', 'LBXSLDSI',
                    'LBXSCH', 'LBXSASSI', 'LBXSIR'], axis = 1, inplace=True)


targets_list = ['DIAGNOSED_DIABETES']

features_dframe = full_data.drop(targets_list, axis=1, inplace=False)

#Print all labels in data
data_cols = list(full_data.columns)
print("All labels in data:\n")
for i, col in enumerate(data_cols):
    print("i = {}, feat = {}".format(i, col))
print("")


#Print all labels in data
feat_cols = list(features_dframe.columns)
print("All labels in model features dataframe:\n")
for i, col in enumerate(feat_cols):
    print("i = {}, feat = {}".format(i, col))
print("")


print("Model features description:\n")
print(features_dframe.describe())
print("")
print("Target features description:\n")
print(full_data[targets_list].describe())


#feature_sets definition
#Compute before hand the number of possible combinations of features used for training
feature_combinatons = combinations(iterable=feat_cols, r=7)
feature_sets = []
for el in feature_combinatons:
    feature_sets.append(list(el))
print("Number of feature combination sets = {}".format(len(feature_sets)))


def testPerformance(full_data, features_list, targets_list, oversample=False):

    # Select model features
    features_df = full_data[features_list]
    features_df = pd.DataFrame(data=features_df, columns=features_list)

    # Select model targets
    targets_df = full_data[targets_list]
    targets_df = pd.DataFrame(data=targets_df, columns=targets_list)

    seed = 7
    np.random.seed(seed)

    #Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, targets_df, test_size=0.33, random_state=seed)

    #SMOTE Oversampling step
    if oversample:

        X_train, y_train = SMOTE(ratio='minority',kind='regular',k_neighbors=3).fit_sample(X_train, y_train)
        X_train = pd.DataFrame(data=X_train, columns=features_df.columns)
        y_train = pd.DataFrame(data=y_train, columns=targets_df.columns)

    #XGBOOST Training Phase

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    predictions_prob_test  = model.predict(X_test)
    predictions_test = [round(value) for value in predictions_prob_test]

    # Calculate ROC curve
    fpr, tpr, dash = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

    # Calculate the AUC
    roc_auc = auc(fpr, tpr)

    accuracy = accuracy_score(y_test, predictions_test)
    acc_pctg = accuracy * 100.0

    return (features_list, roc_auc, acc_pctg)

#feat_sets defintion
#Subselection of feature sets
number_of_test_sets = 20
feat_sets = []
for m in range(number_of_test_sets):
    feat_sets.append(feature_sets[m])

import multiprocessing as mp

#Synchronus thread execution
if False:
    star_time = time.time()
    pool = mp.Pool()
    results = [pool.apply(testPerformance, args=(full_data, x, targets_list )) for x in feat_sets]
    end_time = time.time()
    for rs in results:
        print(rs)
    print("Synchronus Execution_time = {}".format(end_time-star_time))

#Asynchronus thread execution (321 times faster!)
if True:
    stored_results = []
    star_time = time.time()
    pool = mp.Pool()
    results = [pool.apply_async(testPerformance, args=(full_data, x, targets_list )) for x in feat_sets]
    end_time = time.time()
    output = [p.get() for p in results]
    for rs in output:
        stored_results.append(rs)
        print(rs)

    print("Asynchronus Execution_time = {}".format(end_time-star_time))

resd = {'FEATURES':[], "ROC":[], "ACC":[]}
resdf = pd.DataFrame(resd)

for tmp in stored_results:
    resdf = resdf.append(pd.DataFrame({"FEATURES":[tmp[0]], "ROC":[tmp[1]], "ACC":[tmp[2]]}))

print(resdf)