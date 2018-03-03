

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
bio_cols = []
que_cols = []

for col in data_cols:
    if ('LBX' in col):
        bio_cols.append(col)
    else:
        que_cols.append(col)

biochemistry_data = full_data[bio_cols]
questionnaire_data = full_data[que_cols]

if False:
    print(biochemistry_data.describe())
    print("\n---------------------------\n")
    print(questionnaire_data.describe())
print("\nData columns:\n")
if True:
    print("\nBiochemistry Data\n")
    print(biochemistry_data.columns)
    print("\nQuestionnaire Data\n")
    print(questionnaire_data.columns)



#Example: it proves that corrcoef, pearsonr and pointbiserial yields the same correlation
#         coeff values between a continuous and a dichotomous variable
if False:

    pair_df = pd.concat([biochemistry_data.LBXSTR, questionnaire_data.HYPERTENSION], axis=1)
    print(pair_df.describe())

    cols = list(pair_df.columns)
    corr_matrix = np.corrcoef(pair_df[cols].values.T)

    print("\nCorrelation matrix:")
    print(corr_matrix)

    from scipy.stats import pearsonr
    print("\nPearson's corr coeff:")
    print(pearsonr(biochemistry_data.LBXSTR,questionnaire_data.HYPERTENSION))

    from scipy.stats import pointbiserialr
    print("\nPoint biserial corr coeff:")
    print(pointbiserialr(biochemistry_data.LBXSTR,questionnaire_data.HYPERTENSION))



#Correlation measures exploration
if False:
    print("\nCorrelation coefficients:\n")
    import dcor # E-statistics module
    from scipy.stats import pearsonr, spearmanr, kendalltau

    for feat in bio_cols:
        print(feat)

        print("Pearson's = {}".format(pearsonr(biochemistry_data[feat], questionnaire_data["HYPERTENSION"])[0]))
        print("Spearman's = {}".format(spearmanr(biochemistry_data[feat], questionnaire_data["HYPERTENSION"])[0]))
        print("Kendall's Tau = {}\n".format(kendalltau(biochemistry_data[feat], questionnaire_data["HYPERTENSION"])[0]))
        print("Distance Correlation = {}".format(dcor.distance_correlation(biochemistry_data[feat], questionnaire_data["HYPERTENSION"])))
        print("Energy Distance = {}\n".format(dcor.energy_distance(biochemistry_data[feat], questionnaire_data["HYPERTENSION"])))


#Plot corr matrix heatmap
if False:
    import seaborn as sns
    cols = list(biochemistry_data.columns)
    corr_matrix = np.corrcoef(biochemistry_data[cols].values.T)
    print(corr_matrix)
    plt.figure(1, figsize=(12, 18))
    sns.set(font_scale=1.0)
    heat_map = sns.heatmap(corr_matrix, cbar=False, annot=True, square=True, fmt='.2f',
                   annot_kws = {'size': 10}, yticklabels=cols, xticklabels=cols)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.tight_layout()
    plt.show()


import dcor # E-statistics module
import seaborn as sns
cols = list(biochemistry_data.columns)

#version 2
if True:

    container = len(cols)*len(cols)*[0.0]
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            m = i * len(cols) + j
            n = j * len(cols) + i
            container[m] = dcor.distance_correlation(biochemistry_data[cols[i]], biochemistry_data[cols[j]])
            container[n] = container[m]

    distance_corr_matrix = np.reshape(container, (-1, len(cols)))

    plt.figure(1, figsize=(12, 18))
    sns.set(font_scale=1.0)
    heat_map = sns.heatmap(distance_corr_matrix, cbar=False, annot=True, square=True, fmt='.2f',
                   annot_kws = {'size': 10}, yticklabels=cols, xticklabels=cols)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.tight_layout()
    plt.show()

#version 1
if False:
    distance_corr_matrix = np.zeros((len(cols), len(cols)))
    container = []
    for i, col_i in enumerate(cols):
        for j, col_j in enumerate(cols):
            container.append(dcor.distance_correlation(biochemistry_data[col_i], biochemistry_data[col_j]))


    distance_corr_matrix = np.reshape(container, (-1, len(cols)))

    plt.figure(1, figsize=(12, 18))
    sns.set(font_scale=1.0)
    heat_map = sns.heatmap(distance_corr_matrix, cbar=False, annot=True, square=True, fmt='.2f',
                   annot_kws = {'size': 10}, yticklabels=cols, xticklabels=cols)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.tight_layout()
    plt.show()


