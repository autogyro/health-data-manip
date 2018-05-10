
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
def testPerformance(model, full_data, features_list, targets_list, cross_val=True,cv_folds=5, early_stopping_rounds=50):

    # Select model features
    features_df = full_data[features_list]
    features_df = pd.DataFrame(data=features_df, columns=features_list)

    # Select model targets
    targets_df = full_data[targets_list]
    targets_df = pd.DataFrame(data=targets_df, columns=targets_list)

    #configure cross-validation if cross_val=True
    if cross_val:
        xgb_parameters = model.get_xgb_params()
        train_data = xgb.DMatrix(features_df.values, label=targets_df.values)

        cv_results = xgb.cv(xgb_parameters, train_data, num_boost_round=model.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)

        model.set_params(n_estimators=cv_results.shape[0])

    # XGBOOST Training Phase

    # Fit the algorithm on the data
    model.fit(features_df, targets_df, eval_metric='auc')

    #accuracy_score(y_true, y_pred)

    # Predict training set:
    predictions = model.predict(features_df)
    predictions_prob = model.predict_proba(features_df)[:, 1]


    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(targets_df.values, predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(targets_df, predictions_prob))

    # Calculate ROC curve
    fpr, tpr, dash = roc_curve(targets_df, predictions_prob)

    # Calculate the AUC
    roc_auc = auc(fpr, tpr)

    accuracy = accuracy_score(targets_df.values, predictions)
    acc_pctg = accuracy * 100.0

    ax = xgb.plot_importance(model, grid=False)
    plt.yticks(fontsize=6)
    plt.show()

    return (features_list, roc_auc, acc_pctg)


#Testing cross validation

#=====================================================                     =============================================
#----------------------------------------------------- GRIDSEARCH CV TESTS ---------------------------------------------
######################################################                     #############################################

#Default test (simplest case)
if False:
    model_1 = xgb.XGBClassifier()

    opt_feats = ['LBXSBU', 'LBXSCK', 'LBXSCR', 'LBXSGL', 'LBXSKSI', 'BMI', 'HYPERTENSION_ONSET', 'HIGHCHOL_ONSET', 'HIGHCHOL', 'FAMILIAL_DIABETES']
    target_feat = ['DIAGNOSED_DIABETES']

    resultados = testPerformance(model_1, full_data, opt_feats, target_feat, cross_val=True)
    print(resultados)

#Optimization 1
if False:
    # Choose all predictors except target & IDcols
    pred_features = [x for x in full_data.columns if x not in ['DIAGNOSED_DIABETES', 'SEQN']]

    target_feat = ['DIAGNOSED_DIABETES']

    model_2 = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

    resultados = testPerformance(model_2, full_data, pred_features, target_feat, cross_val=True)
    print(resultados)

#Optimization GridSearch CV 1
if False:
    param_test1 = {
        'max_depth': range(3, 10, 1),
        'min_child_weight': range(1, 6, 1),
        'n_estimators': range(10,150,10)
    }
    gsearch1 = GridSearchCV(estimator=xgb.XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5,
                                                    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test1, scoring='roc_auc', n_jobs=8, iid=False, cv=5)


    pred_features = [x for x in full_data.columns if x not in ['DIAGNOSED_DIABETES', 'SEQN']]
    target_feat = 'DIAGNOSED_DIABETES'

    gsearch1.fit(full_data[pred_features], full_data[target_feat])

    print("\nGrid scores:")
    for rs in gsearch1.grid_scores_:
        print(rs)

    print("\nBest params:")
    print(gsearch1.best_params_)

    print("\nBest ROC score:")
    print(gsearch1.best_score_)

    #Best params: {'n_estimators': 40, 'min_child_weight': 2, 'max_depth': 4}


#Tune gamma
if False:
    param_test2 = {'gamma':[i/10.0 for i in range(0,5)]}

    gsearch2 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=40, max_depth=4,
     min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
     objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
     param_grid = param_test2, scoring='roc_auc',n_jobs=8,iid=False, cv=5)


    pred_features = [x for x in full_data.columns if x not in ['DIAGNOSED_DIABETES', 'SEQN']]
    target_feat = 'DIAGNOSED_DIABETES'

    gsearch2.fit(full_data[pred_features], full_data[target_feat])

    print("\nGrid scores:")
    for rs in gsearch2.grid_scores_:
        print(rs)

    print("\nBest params:")
    print(gsearch2.best_params_)

    print("\nBest ROC score:")
    print(gsearch2.best_score_)

#Recalibration first pass
if False:
    # Choose all predictors except target & IDcols
    pred_features = [x for x in full_data.columns if x not in ['DIAGNOSED_DIABETES', 'SEQN']]

    target_feat = ['DIAGNOSED_DIABETES']

    model_3 = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=40,
        max_depth=4,
        min_child_weight=2,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

    resultados = testPerformance(model_3, full_data, pred_features, target_feat, cross_val=True)
    print(resultados)

#Tune subsample and colsample_bytree values first pass
if False:
    param_test3 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    gsearch3 = GridSearchCV(estimator=xgb.XGBClassifier(learning_rate=0.1, n_estimators=40, max_depth=4,
                                                    min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test3, scoring='roc_auc', n_jobs=8, iid=False, cv=5)

    pred_features = [x for x in full_data.columns if x not in ['DIAGNOSED_DIABETES', 'SEQN']]
    target_feat = 'DIAGNOSED_DIABETES'

    gsearch3.fit(full_data[pred_features], full_data[target_feat])

    print("\nGrid scores:")
    for rs in gsearch3.grid_scores_:
        print(rs)

    print("\nBest params:")
    print(gsearch3.best_params_)

    print("\nBest ROC score:")
    print(gsearch3.best_score_)

    #Best params:{'subsample': 0.9, 'colsample_bytree': 0.7}


#Tune subsample and colsample_bytree values second pass
if False:
    param_test4 = {
        'subsample': [i / 100.0 for i in range(80, 100)],
        'colsample_bytree': [i / 100.0 for i in range(60, 100)]
    }
    gsearch4 = GridSearchCV(estimator=xgb.XGBClassifier(learning_rate=0.1, n_estimators=40, max_depth=4,
                                                    min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test4, scoring='roc_auc', n_jobs=8, iid=False, cv=5)

    pred_features = [x for x in full_data.columns if x not in ['DIAGNOSED_DIABETES', 'SEQN']]
    target_feat = 'DIAGNOSED_DIABETES'

    gsearch4.fit(full_data[pred_features], full_data[target_feat])

    print("\nGrid scores:")
    for rs in gsearch4.grid_scores_:
        print(rs)

    print("\nBest params:")
    print(gsearch4.best_params_)

    print("\nBest ROC score:")
    print(gsearch4.best_score_)

    #{'subsample': 0.81, 'colsample_bytree': 0.71}


#Tune regularization parameter alpha
if False:


    param_test5 = {'reg_alpha': [0.13, 0.14, 0.145, 0.15, 0.16]}


    gsearch5 = GridSearchCV(estimator=xgb.XGBClassifier(learning_rate=0.1, n_estimators=40, max_depth=4,
                                                    min_child_weight=2, gamma=0, subsample=0.81, colsample_bytree=0.71,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test5, scoring='roc_auc', n_jobs=8, iid=False, cv=5)

    pred_features = [x for x in full_data.columns if x not in ['DIAGNOSED_DIABETES', 'SEQN']]
    target_feat = 'DIAGNOSED_DIABETES'

    gsearch5.fit(full_data[pred_features], full_data[target_feat])

    print("\nGrid scores:")
    for rs in gsearch5.grid_scores_:
        print(rs)

    print("\nBest params:")
    print(gsearch5.best_params_)

    print("\nBest ROC score:")
    print(gsearch5.best_score_)

    #Best params: {'reg_alpha': 0.14}



#Recalibration second pass
if False:
    # Choose all predictors except target & IDcols
    pred_features = [x for x in full_data.columns if x not in ['DIAGNOSED_DIABETES', 'SEQN']]

    target_feat = ['DIAGNOSED_DIABETES']

    model_4 = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=40,
        max_depth=4,
        min_child_weight=2,
        gamma=0,
        subsample=0.81,
        colsample_bytree=0.71,
        reg_alpha=0.14,
        objective='binary:logistic',
        nthread=8,
        scale_pos_weight=1,
        seed=27)

    resultados = testPerformance(model_4, full_data, pred_features, target_feat, cross_val=True)
    print(resultados)


#Tuning learning rate (we might decide to stop tuning here)
if True:
    # Choose all predictors except target & IDcols
    pred_features = [x for x in full_data.columns if x not in ['DIAGNOSED_DIABETES', 'SEQN']]

    target_feat = ['DIAGNOSED_DIABETES']

    model_5 = xgb.XGBClassifier(
        learning_rate=0.01,
        n_estimators=400,
        max_depth=4,
        min_child_weight=2,
        gamma=0,
        subsample=0.81,
        colsample_bytree=0.71,
        reg_alpha=0.14,
        objective='binary:logistic',
        nthread=8,
        scale_pos_weight=1,
        seed=27)

    resultados = testPerformance(model_5, full_data, pred_features, target_feat, cross_val=True)
    print(resultados)



# ============================================COMBINATIONS DIRECT OPTIMIZATION =========================================
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

