
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

bio_cols = ['LBXSGL']
#bio_cols = ['LBXSGL', 'LBXSOSSI']


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
    from xgboost import XGBClassifier

    from sklearn.metrics import fbeta_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve, auc

    seed = 7
    np.random.seed(seed)

    X0 = model_features.values
    Y0 = model_targets.values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X0, Y0, test_size=0.33, random_state=seed)

    #=============================================================
    #-------------------- Oversampling step ----------------------

    from imblearn.over_sampling import SMOTE
    from imblearn.over_sampling import RandomOverSampler

    #Oversampling over entire dataset
    if False:
        X, Y = SMOTE(ratio='minority').fit_sample(X0, Y0)

    # SMOTE Oversampling over the training dataset
    if False:
        X_train, y_train = SMOTE(ratio='minority',kind='regular',k_neighbors=3).fit_sample(X_train, y_train)

    # Random Oversampling over the training dataset
    if False:
        X_train, y_train = RandomOverSampler(ratio='minority',random_state=0).fit_sample(X_train, y_train)


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

    pred_train_df = pd.concat([pd.DataFrame(predictions_prob_train, columns=['PROB']), pd.DataFrame(predictions_train, columns=['Y_PRED_TRAIN'])], axis=1)
    pred_test_df  = pd.concat([pd.DataFrame(predictions_prob_test, columns=['PROB']), pd.DataFrame(predictions_test, columns=['Y_PRED_TEST'])], axis=1)

    if False:
        print("Predictions Preview:\n")
        txt.headcounts(pred_train_df, 100)
        txt.headcounts(pred_test_df,100)

    #acc_train = accuracy_score(y_train, predictions_train)
    #acc_test = accuracy_score(y_test, predictions_test)

    #beta = 0.5

    #fb_train = fbeta_score(y_train, predictions_train, beta=beta)
    #fb_test = fbeta_score(y_test, predictions_test, beta=beta)

    #f1_train = f1_score(y_train, predictions_train)
    #f1_test = f1_score(y_test, predictions_test)

    #roc_auc_train = roc_auc_score(y_train, predictions_train, average='micro')
    #roc_auc_test = roc_auc_score(y_test, predictions_test, average='micro')

    from sklearn.metrics import confusion_matrix

    CM = confusion_matrix(y_test, predictions_test)
    CML = np.array([['TN', 'FP'], ['FN', 'TP']])

    #print("acc_train = {}, acc_test ={}".format(acc_train, acc_test))
    print("Confusion Matrix:\n{}\n\n {} \n".format(CML, CM))
    #print("f1_train = {}, f1_test ={}".format(f1_train, f1_test))
    #print("fbeta_train = {}, fbeta_test ={}".format(fb_train, fb_test))
    #print("ROC_AUC_train = {}, ROC_AUC_test ={}".format(roc_auc_train, roc_auc_test))


    if False:
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

#################################################################################################################
################################# Neural Network Model ############################################
#################################################################################################################


if False:

    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    from keras.layers import Dropout
    from keras.constraints import maxnorm
    from keras.optimizers import SGD
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV


    from sklearn.metrics import fbeta_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve, auc

    #------Keras custom metrics--------
    import keras.backend as K

    #----not working.....
    def k_custom_auc(y_true, y_pred):

        fpr, tpr, dash = roc_curve(y_true, y_pred)
        score = auc(fpr, tpr)
        return K.variable(score)

    #----------------------------------

    seed = 7
    np.random.seed(seed)

    X = model_features.values
    Y = model_targets.values

    print(X.shape)
    print(Y.shape)

    sys.exit()


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)


    #//////// NN Models ////////////////////////////////////////////////////////////////////////////////


    def initial_model():

        model = Sequential()
        model.add(Dense(64, input_dim=features_num, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


    def second_model():
        # create model
        model = Sequential()
        model.add(Dense(64, input_dim=features_num, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Dense(32, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
        #model.add(Dropout(0.2))
        model.add(Dense(16, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
        #model.add(Dropout(0.2))
        model.add(Dense(18, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
        #model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        # Compile model
        sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    #//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # ------------ USER_DEFINED SKLEARN SCORING FUNCTIONS -------------
    # To be used in required subroutines below
    # ---------------------------------------------------------

    def performance_metric_auc(y, y_pred):

        fpr, tpr, dash = roc_curve(y, y_pred)
        score = auc(fpr, tpr)

        return score



    #Implement unoptimized models
    #----------------------------
    if False:
        model = second_model()

        num_epochs = 100
        batch_size = 32

        hm = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=num_epochs, batch_size=batch_size)
        gut.plot_KerasHistory_metrics(hm, 'nhanes_keras_model_metrics')

        scores = model.evaluate(X, Y)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

        prob_train = model.predict(X_train)
        predictions_train = [round(x[0]) for x in prob_train]

        prob_test = model.predict(X_test)
        predictions_test = [round(x[0]) for x in prob_test]

        pred_train_df = pd.concat([pd.DataFrame(prob_train, columns=['PROB']), pd.DataFrame(predictions_train, columns=['Y_PRED_TRAIN'])], axis=1)
        pred_test_df  = pd.concat([pd.DataFrame(prob_test, columns=['PROB']), pd.DataFrame(predictions_test, columns=['Y_PRED_TEST'])], axis=1)

    #---------------------------------------------------------------------------------
    ############## Perform GridSearchCV (Optimize Epochs, Batch Size) ################
    #---------------------------------------------------------------------------------
    if True:

        seed = 7
        np.random.seed(seed)


        #----------------------------------------------------------------------
        #select one of the models defined above and insert it as hyperparameter
        #model = KerasClassifier(build_fn=second_model, verbose=0)
        model = KerasClassifier(build_fn=initial_model, verbose=0)
        #-----------------------------------------------------------------------

        # define the grid search parameters

        batch_size = [10, 20, 40]   #batch_size = [10, 20, 40, 60, 80, 100]
        epochs = [10, 50,100]       #epochs = [10, 50, 100]

        param_grid = dict(batch_size=batch_size, epochs=epochs)

        #----------------- scoring functions
        scoring_auc = make_scorer(performance_metric_auc, needs_proba=False)


        ######
        #grid = GridSearchCV(estimator=model, scoring="roc_auc", param_grid=param_grid, n_jobs=1)
        grid = GridSearchCV(estimator=model, scoring=scoring_auc, param_grid=param_grid, n_jobs=1)
        ######
        grid_result = grid.fit(X, Y)

        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

    #(optional) print preview of predictions arrays
    if False:
        print("Predictions Preview:\n")
        txt.headcounts(pred_train_df)
        txt.headcounts(pred_test_df)

    #(optional) print predicted probabilites  on test set
    if False:
        print(model.predict_proba(X_test))
        print(model.predict_proba(X_test).shape)
        sys.exit()

    #Individual performance metrics:

    if False:
        #Confusion matrix
        if True:
            from sklearn.metrics import confusion_matrix
            CM = confusion_matrix(y_test, predictions_test)
            CML = np.array([['TN', 'FP'], ['FN', 'TP']])
            print("Confusion Matrix:\n{}\n\n {} \n".format(CML, CM))


        #Calculate false, true positive rates (fpr, tpr), and typing monkey roc line (dash)
        #Optionally save the datapoints to csv
        if False:
            fpr, tpr, dash = roc_curve(y_test, model.predict_proba(X_test))
            if True:
                rocxg_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'dash': dash})
                rocxg_df.to_csv('roc_curve_cnn.csv')

        #AUC score and accuracy
        if True:
            roc_auc = auc(fpr, tpr)
            print('ROC AUC: %0.2f' % roc_auc)
            accuracy = accuracy_score(y_test, predictions_test)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))


        #accuracy score
        if True:
            acc_train = accuracy_score(y_train, predictions_train)
            acc_test = accuracy_score(y_test, predictions_test)
            print("acc_train = {}, acc_test ={}".format(acc_train, acc_test))


        #beta score optional
        if False:
            beta = 0.5
            fb_train = fbeta_score(y_train, predictions_train, beta=beta)
            fb_test = fbeta_score(y_test, predictions_test, beta=beta)
            f1_train = f1_score(y_train, predictions_train)
            f1_test = f1_score(y_test, predictions_test)

            print("f1_train = {}, f1_test ={}".format(f1_train, f1_test))
            print("fbeta_train = {}, fbeta_test ={}".format(fb_train, fb_test))

        #roc auc (not confuese with auc)
        if False:
            roc_auc_train = roc_auc_score(y_train, predictions_train, average='micro')
            roc_auc_test = roc_auc_score(y_test, predictions_test, average='micro')
            print("ROC_AUC_train = {}, ROC_AUC_test ={}".format(roc_auc_train, roc_auc_test))

    #--------------------------------------------------------------------
    # Plot of a ROC curve for a specific class
    #--------------------------------------------------------------------
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

    tf.Session().close()


