
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


from sklearn.model_selection import train_test_split

def initial_model(neurons=20):
    model = Sequential()
    model.add(Dense(neurons, input_dim=features_num, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def pf_metric_auc(y, y_pred):
    fpr, tpr, dash = roc_curve(y, y_pred)
    score = auc(fpr, tpr)
    return score

scoring_auc = make_scorer(pf_metric_auc, needs_proba=False)

seed = 7
np.random.seed(seed)

X = model_features.values
Y = model_targets.values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

#-----Model selection-------------
#Bare Keras
if True:
    Keras_model = initial_model(neurons=20)
    scores = Keras_model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (Keras_model.metrics_names[1], scores[1] * 100))

    num_epochs = 20
    batch_size = 10
    hm = Keras_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, batch_size=batch_size)
    gut.plot_KerasHistory_metrics(hm, 'nhanes_keras_model_metrics')
    plt.show()

#SKLearn
if False:
    SKlearn_model = KerasClassifier(build_fn=initial_model, neurons=20, verbose=0)
#-----End Model selection----------




tf.Session().close()