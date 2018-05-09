
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from sklearn import cross_validation, metrics   #Additional scklearn functions
#from sklearn.grid_search import GridSearchCV    #Perforing grid search
from sklearn.model_selection import GridSearchCV

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


#Model fit function
def testPerformance(model, full_data, features_list, targets_list, cross_val=True):

    # Select model features
    features_df = full_data[features_list]
    features_df = pd.DataFrame(data=features_df, columns=features_list)

    # Select model targets
    targets_df = full_data[targets_list]
    targets_df = pd.DataFrame(data=targets_df, columns=targets_list)

    #configure cross-validation if cross_val=True
    if cross_val:
        xgb_parameters = alg.get_xgb_params()
        train_data = xgb.DMatrix(features_df.values, label=targets_df.values)

        cv_results = xgb.cv(xgb_parameters, train_data, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)

        model.set_params(n_estimators=cv_results.shape[0])

    # XGBOOST Training Phase

    # Fit the algorithm on the data
    model.fit(features_df, targets_df, eval_metric='auc')

    #accuracy_score(y_true, y_pred)

    # Predict training set:
    predictions = model.predict(features_df)
    predictions_prob= alg.predict_proba(features_df)[:, 1]

    # Print model report:
    print
    "\nModel Report"
    print
    "Accuracy : %.4g" % metrics.accuracy_score(targets_df.values, predictions)
    print
    "AUC Score (Train): %f" % metrics.roc_auc_score(targets_df, predictions_prob)

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


    # Calculate ROC curve
    fpr, tpr, dash = roc_curve(features_df, model.predict_proba(targets_df.values)[:, 1])

    # Calculate the AUC
    roc_auc = auc(fpr, tpr)

    accuracy = accuracy_score(targets_df.values, predictions)
    acc_pctg = accuracy * 100.0

    return (features_list, roc_auc, acc_pctg)


#Define model to evaluate
model = xgb.XGBClassifier()



#feature_sets definition
#Compute before hand the number of possible combinations of features used for training
if False:
    feature_combinatons = combinations(iterable=feat_cols, r=10)
    feature_sets = []
    for el in feature_combinatons:
        feature_sets.append(list(el))
    print("Number of feature combination sets = {}".format(len(feature_sets)))

    # feat_sets defintion
    # Subselection of feature sets
    if False:
        number_of_test_sets = 10
    if True:
        number_of_test_sets = 20
    if False:
        number_of_test_sets = len(feature_sets)

    feat_sets = []
    for m in range(number_of_test_sets):
        feat_sets.append(feature_sets[m])


#Evaluate model on multiple feature combinations
#Asynchronus thread execution (much faster!)
if False:
    import multiprocessing as mp

    stored_results = []
    star_time = time.time()
    pool = mp.Pool()
    results = [pool.apply_async(testPerformance, args=(full_data, x, targets_list )) for x in feat_sets]
    end_time = time.time()
    output = [p.get() for p in results]
    for rs in output:
        stored_results.append(rs)
        if False:
            print(rs)

    print("Asynchronus Execution_time = {}".format(end_time-star_time))

    resd = {'FEATURES':[], "ROC":[], "ACC":[]}
    resdf = pd.DataFrame(resd)

    for tmp in stored_results:
        resdf = resdf.append(pd.DataFrame({"FEATURES":[tmp[0]], "ROC":[tmp[1]], "ACC":[tmp[2]]}))

    #Sort resdf dataframe by ROC values
    resdf.sort_values(by=['ROC'], ascending=False, inplace=True)
    resdf.reset_index(level='int', inplace=True)

    #Save resdf DF to CSV File
    resdf.to_csv('feature_combinations_results.csv')

    print("\nMaximum ROC value in data:\n")
    max_record_roc = resdf.loc[resdf['ROC'].idxmax()]
    print(max_record_roc)
    print(max_record_roc["FEATURES"])

    print("\nMaximum Accuracy value in data:\n")
    max_record_acc = resdf.loc[resdf['ACC'].idxmax()]
    print(max_record_acc)
    print(max_record_acc["FEATURES"])

