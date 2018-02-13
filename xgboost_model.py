
import sys
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#import local modules
from importlib.machinery import SourceFileLoader
mod_path = "/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/nhanes2013-2104/"
txt = SourceFileLoader("text_utils", mod_path+"text_utils.py").load_module()
gut = SourceFileLoader("graph_utils", mod_path+"graph_utils.py").load_module()

metrics = SourceFileLoader("custom_metrics", mod_path+"custom_metrics.py").load_module()


#Path to datasets folders

datasets_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/'
project_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/'

full_data = pd.read_csv(project_path + 'nhanes_2013_2014_full_data.csv',index_col=0)

data_cols = list(full_data.columns)
bio_cols = []
que_cols = []

for col in data_cols:
    if ('LBX' in col):
        bio_cols.append(col)
    else:
        que_cols.append(col)

biochemistry_data = full_data[bio_cols]

if True:
    print(biochemistry_data.describe())


##################################### INPUT ###################################
#Model input features
model_features = pd.concat([biochemistry_data, full_data.BMI], axis=1)
model_features = pd.concat([model_features, full_data.AGE], axis=1)
model_features = pd.concat([model_features, full_data.SMOKING], axis=1)
model_features = pd.concat([model_features, full_data.ALCOHOL_NUM], axis=1)
model_features = pd.concat([model_features, full_data.FAMILIAL_DIABETES], axis=1)
model_features = pd.concat([model_features, full_data.GENDER_Male], axis=1)
model_features = pd.concat([model_features, full_data.GENDER_Female], axis=1)
model_features = pd.concat([model_features, full_data.ETHNICITY_White], axis=1)
model_features = pd.concat([model_features, full_data.ETHNICITY_Black], axis=1)
model_features = pd.concat([model_features, full_data.ETHNICITY_Hispanic], axis=1)
model_features = pd.concat([model_features, full_data.ETHNICITY_Asian], axis=1)

#################################### TARGET #####################################
#Target for prediction/classification

model_targets = full_data.DIAGNOSED_PREDIABETES
#model_targets = full_data.HIGHCHOL
#model_targets = full_data.RISK_DIABETES
#model_targets = full_data.HYPERTENSION

features_num = len(model_features.columns)
print("Input features:\n")
print(model_features.columns)
print("No. of input features: {}".format(features_num))


#Initial test
if True:
    from xgboost import XGBClassifier

    from sklearn.metrics import fbeta_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve, auc

    seed = 7
    np.random.seed(seed)

    X = model_features.values
    Y = model_targets.values

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

    model = XGBClassifier()
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

    acc_train = accuracy_score(y_train, predictions_train)
    acc_test = accuracy_score(y_test, predictions_test)
    print("acc_train = {}, acc_test ={}".format(acc_train, acc_test))

    CM = confusion_matrix(y_test, predictions_test)
    CML = np.array([['TN', 'FP'], ['FN', 'TP']])
    print("Confusion Matrix:\n{}\n\n {} \n".format(CML, CM))

    pred_train_df = pd.concat([pd.DataFrame(predictions_prob_train, columns=['PROB']), pd.DataFrame(predictions_train, columns=['Y_PRED_TRAIN'])], axis=1)
    pred_test_df  = pd.concat([pd.DataFrame(predictions_prob_test, columns=['PROB']), pd.DataFrame(predictions_test, columns=['Y_PRED_TEST'])], axis=1)

    #predictions preview
    if False:
        print("Predictions Preview:\n")
        txt.headcounts(pred_train_df, 100)
        txt.headcounts(pred_test_df,100)

    # Plot of a ROC curve for a specific class
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



    #---------------------------------------------------------------
    #other metrics
    from sklearn.metrics import confusion_matrix
    if False:
        beta = 0.5

        fb_train = fbeta_score(y_train, predictions_train, beta=beta)
        fb_test = fbeta_score(y_test, predictions_test, beta=beta)

        f1_train = f1_score(y_train, predictions_train)
        f1_test = f1_score(y_test, predictions_test)

        roc_auc_train = roc_auc_score(y_train, predictions_train, average='micro')
        roc_auc_test = roc_auc_score(y_test, predictions_test, average='micro')

        print("f1_train = {}, f1_test ={}".format(f1_train, f1_test))
        print("fbeta_train = {}, fbeta_test ={}".format(fb_train, fb_test))
        print("ROC_AUC_train = {}, ROC_AUC_test ={}".format(roc_auc_train, roc_auc_test))








