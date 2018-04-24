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

from datetime import datetime





mod_path = "/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/nhanes2013-2104/"
txt = SourceFileLoader("text_utils", mod_path+"text_utils.py").load_module()
gut = SourceFileLoader("graph_utils", mod_path+"graph_utils.py").load_module()

#Path to datasets folders

datasets_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/'
project_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/'

full_data = pd.read_csv(project_path + 'nhanes_2013_2014_full_data.csv',index_col=0)


full_data.drop(['ETHNICITY_Other', 'GENDER_Male', 'GENDER_Female','DIAGNOSED_PREDIABETES',
                'BREATH_SHORTNESS', 'CHEST_PAIN_30MIN', 'CHEST_DISCOMFORT', 'RISK_DIABETES',
                'FAST_FOOD', 'NOTHOME_FOOD', 'INCOME_LEVEL','ETHNICITY_White', 'ETHNICITY_Black',
                'ETHNICITY_Hispanic', 'ETHNICITY_Asian', 'SMOKING'], axis = 1, inplace=True)

full_data.drop(['LBXSTP', 'LBXSPH', 'LBXSC3SI', 'LBXSCA', 'LBXSLDSI', 'LBXSCK',
                'LBXSCH', 'LBXSUA', 'LBXSASSI', 'LBXSIR'], axis = 1, inplace=True)


#Print all labels in data
data_cols = list(full_data.columns)
print("All labels in data:\n")
for i, col in enumerate(data_cols):
    print("i = {}, feat = {}".format(i, col))
print("")

#Select model features
model_features = full_data.drop(['DIAGNOSED_DIABETES'], axis = 1, inplace=False)

#Select model targets
model_targets = full_data.DIAGNOSED_DIABETES
model_targets = pd.DataFrame(data=model_targets, columns=['DIAGNOSED_DIABETES'])

#Print all labels in data
feat_cols = list(model_features.columns)
print("All labels in model features dataframe:\n")
for i, col in enumerate(feat_cols):
    print("i = {}, feat = {}".format(i, col))
print("")


print("Model features description:\n")
print(model_features.describe())
print("")
print("Target features description:\n")
print(model_targets.describe())


feature_combinatons = combinations(iterable=feat_cols, r=7)
feature_sets = []
for el in feature_combinatons:
    feature_sets.append(list(el))
print("Number of feature combination sets = {}".format(len(feature_sets)))


def testPerformance(features, targets, oversample=False):

    np.random.seed(datetime.now())

    #Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.33, random_state=seed)

    #SMOTE Oversampling step
    if oversample:

        X_train, y_train = SMOTE(ratio='minority',kind='regular',k_neighbors=3).fit_sample(X_train, y_train)
        X_train = pd.DataFrame(data=X_train, columns=features.columns)
        y_train = pd.DataFrame(data=y_train, columns=targets.columns)

    #XGBOOST Training Phase

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # Calculate ROC curve
    fpr, tpr, dash = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

    # Calculate the AUC
    roc_auc = auc(fpr, tpr)

    accuracy = accuracy_score(y_test, predictions_test)
    acc_pctg = accuracy * 100.0

    return (features, roc_auc, acc_pctg)



