
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

full_data = pd.read_csv(project_path + 'nhanes_2013_2014_full_data.csv',index_col=0)

data_cols = list(full_data.columns)
print("All labels in data:\n")
print(data_cols)
print("")

# Using all column labels in data
if False:
    bio_cols = []
    que_cols = []

    for col in data_cols:
        if ('LBX' in col):
            bio_cols.append(col)
        else:
            que_cols.append(col)

#Selected column labels (see heat maps associations)

#bio_cols = ['LBXSGL']
bio_cols = ['LBXSGL', 'LBXSOSSI']
#bio_cols = ['LBXSOSSI', 'LBXSGL']


biochemistry_data = full_data[bio_cols]

if True:
    print(biochemistry_data.describe())


if True:
    ##################################### INPUT ###################################
    #Model input features
    model_features = pd.concat([biochemistry_data, full_data.BMI], axis=1)
    model_features = pd.concat([model_features, full_data.AGE], axis=1)
    model_features = pd.concat([model_features, full_data.HYPERTENSION_ONSET], axis=1)
    model_features = pd.concat([model_features, full_data.HYPERTENSION], axis=1)
    model_features = pd.concat([model_features, full_data.HIGHCHOL_ONSET], axis=1)
    model_features = pd.concat([model_features, full_data.HIGHCHOL], axis=1)
    model_features = pd.concat([model_features, full_data.SMOKING], axis=1)
    model_features = pd.concat([model_features, full_data.ALCOHOL_NUM], axis=1)
    model_features = pd.concat([model_features, full_data.FAMILIAL_DIABETES], axis=1)
    model_features = pd.concat([model_features, full_data.GENDER_Male], axis=1)
    model_features = pd.concat([model_features, full_data.GENDER_Female], axis=1)
    model_features = pd.concat([model_features, full_data.ETHNICITY_White], axis=1)
    model_features = pd.concat([model_features, full_data.ETHNICITY_Black], axis=1)
    #model_features = pd.concat([model_features, full_data.ETHNICITY_Hispanic], axis=1)
    #model_features = pd.concat([model_features, full_data.ETHNICITY_Asian], axis=1)



    #################################### TARGET #####################################
    #Target for prediction/classification


    model_targets = full_data.DIAGNOSED_DIABETES
    model_targets = pd.DataFrame(data=model_targets, columns=['DIAGNOSED_DIABETES'])

    #model_targets = full_data.DIAGNOSED_PREDIABETES
    #model_targets = full_data.HIGHCHOL
    #model_targets = full_data.RISK_DIABETES
    #model_targets = full_data.HYPERTENSION

    features_num = len(model_features.columns)
    print("Input features:\n")
    print(model_features.columns)
    print("No. of input features: {}".format(features_num))


######################################################################################################
################################# XGBOOST Model ######################################################
######################################################################################################

if True:

    from sklearn.metrics import fbeta_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve, auc

    import xgboost as xgb

    seed = 7
    np.random.seed(seed)

    from sklearn.model_selection import train_test_split

    #Split on dataframes
    if True:
        X_train, X_test, y_train, y_test = train_test_split(
            model_features, model_targets, test_size=0.33, random_state=seed)

    #Split on numpy arrays
    if False:
        X_train, X_test, y_train, y_test = train_test_split(
            model_features.values, model_targets.values, test_size=0.33, random_state=seed)


    #=============================================================
    #-------------------- Oversampling step ----------------------
    if False:

        from imblearn.over_sampling import SMOTE
        from imblearn.over_sampling import RandomOverSampler

        # SMOTE Oversampling over the training dataset
        if True:
            X_train, y_train = SMOTE(ratio='minority',kind='regular',k_neighbors=3).fit_sample(X_train, y_train)

        # Random Oversampling over the training dataset
        if False:
            X_train, y_train = RandomOverSampler(ratio='minority',random_state=0).fit_sample(X_train, y_train)

        #Necessary step as smote removes label names
        X_train = pd.DataFrame(data=X_train, columns=model_features.columns)
        y_train = pd.DataFrame(data=y_train, columns=model_targets.columns)

    #XGBOOST Training Phase

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    predictions_prob_test  = model.predict(X_test)
    predictions_test = [round(value) for value in predictions_prob_test]

    predictions_prob_train = model.predict(X_train)
    predictions_train = [round(value) for value in predictions_prob_train]

    #Short test compare to CNN case
    if False:
        print(model.predict_proba(X_test))
        print(model.predict_proba(X_test).shape)
        sys.exit()

    ############################# ROC_CURVE_XGBOOST ############################################
    fpr, tpr, dash = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    rocxg_df = pd.DataFrame({'fpr':fpr, 'tpr': tpr, 'dash': dash})
    rocxg_df.to_csv('roc_curve_xgboost.csv')
    ############################################################################################


    #--------------------------------------
    # Calculate the AUC
    roc_auc = auc(fpr, tpr)
    print('ROC AUC: %0.2f' % roc_auc)
    #--------------------------------------

    accuracy = accuracy_score(y_test, predictions_test)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    pred_train_df = pd.concat([pd.DataFrame(predictions_prob_train, columns=['PROB']), pd.DataFrame(predictions_train, columns=['Y_PRED_TRAIN'])], axis=1)
    pred_test_df  = pd.concat([pd.DataFrame(predictions_prob_test, columns=['PROB']), pd.DataFrame(predictions_test, columns=['Y_PRED_TEST'])], axis=1)

    if False:
        print("Predictions Preview:\n")
        txt.headcounts(pred_train_df, 100)
        txt.headcounts(pred_test_df,100)


    from sklearn.metrics import confusion_matrix

    CM = confusion_matrix(y_test, predictions_test)
    CML = np.array([['TN', 'FP'], ['FN', 'TP']])

    print("Confusion Matrix:\n{}\n\n {} \n".format(CML, CM))

    #==================================== Feature importances ======================================
    #Print feature importances
    print(model.feature_importances_)

    #Bar plot of feature importances
    if True:
        xgb.plot_importance(model,grid=False)
        plt.show()

        #output: f0, f1, f2..... refer to the first, second, third.... feature in the df.columns list
        #in that same order

    #Plot of a ROC curve for a specific class
    if False:
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Hypertension Risk Prediction')
        plt.legend(loc="lower right")
        plt.show()

