

import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#import local modules
from importlib.machinery import SourceFileLoader
mod_path = "/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/report/development/utilities/"
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



strong_biofeatures = ['LBXSGL', 'LBXSTR', 'LBXSOSSI', 'LBXSBU', 'LBXSGTSI', 'LBXSCR', 'LBXSUA', 'LBXSLDSI', 'LBXSCH',
                      'LBXSATSI', 'LBXSGB', 'LBXSAPSI']

strong_qfeatures = ['HIGHCHOL', 'HYPERTENSION', 'CHEST_DISCOMFORT', 'CHEST_PAIN_30MIN', 'DIAGNOSED_DIABETES']


from scipy.stats import pointbiserialr
import operator

max_corrs = []
for q_feat in strong_qfeatures:
    d={}
    for b_feat in strong_biofeatures:
        corr = pointbiserialr(questionnaire_data[q_feat], biochemistry_data[b_feat])[0]
        if (corr >= 0.1):
            d[b_feat] = np.round(corr,2)

    sorted_d = sorted(d.items(), key=operator.itemgetter(1))

    if (len(sorted_d) == 0):
        max_assoc = None
    else:
        max_assoc = sorted_d[-1]

    t= (q_feat, max_assoc)
    max_corrs.append(t)

print(max_corrs)






# Compute point biserial correlation among all biochemistry features and one questionnaire features
if False:

    from scipy.stats import pointbiserialr

    d = {}

    #q_feat = 'HIGHCHOL'
    #q_feat = 'HIGHCHOL_ONSET'
    #q_feat = 'HYPERTENSION'
    #q_feat = 'CHEST_DISCOMFORT'
    #q_feat = 'CHEST_PAIN_30MIN'
    q_feat = 'DIAGNOSED_DIABETES'

    for b_feat in biochemistry_data.columns:
        corr = pointbiserialr(questionnaire_data[q_feat], biochemistry_data[b_feat])[0]
        if (corr >= 0.1):
            d[b_feat] = np.round(corr,2)

    print("..")
    print(d)

    import operator
    sorted_d = sorted(d.items(), key=operator.itemgetter(1))
    print("..")
    print(sorted_d)


    print("\nMax and Minimum:\n")
    print(sorted_d[0])
    print(sorted_d[-1])



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
    print(pointbiserialr(biochemistry_data.LBXSTR,questionnaire_data.HYPERTENSION)[0])



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


#Plot correlation matrix using Pearson's correlation measure
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


#Plot correlation matrix using Spearman's correlation measure
if False:
    from scipy.stats import spearmanr
    import seaborn as sns

    cols = list(biochemistry_data.columns)

    container = len(cols)*len(cols)*[0.0]
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            m = i * len(cols) + j
            n = j * len(cols) + i
            container[m] = spearmanr(biochemistry_data[cols[i]], biochemistry_data[cols[j]])
            container[n] = container[m]

    spearmans_corr_matrix = np.reshape(container, (-1, len(cols)))

    plt.figure(1, figsize=(12, 18))
    sns.set(font_scale=1.0)
    heat_map = sns.heatmap(spearmans_corr_matrix, cbar=False, annot=True, square=True, fmt='.2f',
                   annot_kws = {'size': 10}, yticklabels=cols, xticklabels=cols)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.tight_layout()
    plt.show()




#Plot correlation matrix using distance correlation measure
if False:
    import dcor  # E-statistics module
    import seaborn as sns

    cols = list(biochemistry_data.columns)

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



################################################################################################################





if False:

    from scipy.stats import pointbiserialr
    corr = pointbiserialr(questionnaire_data['HIGHCHOL'], biochemistry_data['LBXSGL'])[0]
    corr = np.round(bi_corr, 2)
    print(corr)



#Plot correlation map between cardiovascular-related features and biochemistry data
if False:

    cardio_features = ['HYPERTENSION_ONSET', 'HIGHCHOL_ONSET', 'HIGHCHOL', 'HYPERTENSION', 'CHEST_DISCOMFORT',
                       'CHEST_PAIN_30MIN',
                       'BREATH_SHORTNESS', 'DIAGNOSED_DIABETES', 'DIAGNOSED_PREDIABETES', 'RISK_DIABETES']

    cardio_biochem_df = pd.concat([questionnaire_data[cardio_features], biochemistry_data], axis=1)

    from scipy.stats import pointbiserialr

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.reset_orig()

    cols_bis = list(cardio_biochem_df.columns)

    container_bis = len(cols_bis)*len(cols_bis)*[0.0]
    for i in range(len(cols_bis)):
        for j in range(i, len(cols_bis)):
            m = i * len(cols_bis) + j
            n = j * len(cols_bis) + i
            container_bis[m] = pointbiserialr(cardio_biochem_df[cols_bis[i]], cardio_biochem_df[cols_bis[j]])[0]
            container_bis[n] = container_bis[m]

    biserial_corr_matrix = np.reshape(container_bis, (-1, len(cols_bis)))

    plt.figure(1, figsize=(10, 8))
    sns.set(font_scale=0.7)
    heat_map_bis = sns.heatmap(biserial_corr_matrix, cbar=False, annot=True, square=True, fmt='.2f',
                    annot_kws = {'size': 6}, yticklabels=cols_bis, xticklabels=cols_bis)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.tight_layout()
    plt.show()


#Plot correlation map between selected NON-cardiovasculare related features and biochemistry data
if False:

    non_cardio_features = ['AGE', 'ALCOHOL_NUM', 'SMOKING', 'BMI', 'NOTHOME_FOOD', 'FAST_FOOD', 'GENDER_Female',
                           'GENDER_Male']

    non_cardio_biochem_df = pd.concat([questionnaire_data[non_cardio_features], biochemistry_data], axis=1)

    from scipy.stats import pointbiserialr

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.reset_orig()

    cols_bis = list(non_cardio_biochem_df.columns)

    container_bis = len(cols_bis)*len(cols_bis)*[0.0]
    for i in range(len(cols_bis)):
        for j in range(i, len(cols_bis)):
            m = i * len(cols_bis) + j
            n = j * len(cols_bis) + i
            container_bis[m] = pointbiserialr(non_cardio_biochem_df[cols_bis[i]], non_cardio_biochem_df[cols_bis[j]])[0]
            container_bis[n] = container_bis[m]

    biserial_corr_matrix = np.reshape(container_bis, (-1, len(cols_bis)))

    plt.figure(1, figsize=(10, 8))
    sns.set(font_scale=0.7)
    heat_map_bis = sns.heatmap(biserial_corr_matrix, cbar=False, annot=True, square=True, fmt='.2f',
                    annot_kws = {'size': 6}, yticklabels=cols_bis, xticklabels=cols_bis)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.tight_layout()
    plt.show()


#Plot correlation map between questionnaire related features
if False:
    from scipy.stats import pointbiserialr

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.reset_orig()

    cols_bis = list(questionnaire_data.columns)

    container_bis = len(cols_bis) * len(cols_bis) * [0.0]
    for i in range(len(cols_bis)):
        for j in range(i, len(cols_bis)):
            m = i * len(cols_bis) + j
            n = j * len(cols_bis) + i
            container_bis[m] = pointbiserialr(questionnaire_data[cols_bis[i]], questionnaire_data[cols_bis[j]])[0]
            container_bis[n] = container_bis[m]

    biserial_corr_matrix = np.reshape(container_bis, (-1, len(cols_bis)))

    plt.figure(1, figsize=(10, 8))
    sns.set(font_scale=0.7)
    heat_map_bis = sns.heatmap(biserial_corr_matrix, cbar=False, annot=True, square=True, fmt='.2f',
                               annot_kws={'size': 6}, yticklabels=cols_bis, xticklabels=cols_bis)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.tight_layout()
    plt.show()



#Histogram and density plots of LBXSGL glucose levels prevalence in patients diagnosed/not-diagnosed with diabetes
if False:

    df = pd.concat([questionnaire_data['DIAGNOSED_DIABETES'], biochemistry_data['LBXSGL']], axis=1)
    dflm_p = df[(df['DIAGNOSED_DIABETES'] == 1)]
    dflm_n = df[(df['DIAGNOSED_DIABETES'] == 0)]

    plt.figure(1, figsize=(10, 8))

    #plt.suptitle('Glucose Level Statistics for Patients with/without Diagnosed Diabetes', fontsize=12)

    plt.subplot(2, 2, 1)
    plt.title('Patients with diagnosed diabetes', fontsize=10)
    dflm_p['LBXSGL'].plot(kind='hist', histtype='stepfilled', alpha=0.5, bins=50)
    #plt.xlabel('LBXSGL - Glucose (mg/dL)')

    plt.subplot(2, 2, 2)
    plt.title('Patients NOT diagnosed with diabetes', fontsize=10)
    #plt.xlabel('LBXSGL - Glucose (mg/dL)')
    dflm_n['LBXSGL'].plot(kind='hist', histtype='stepfilled', alpha=0.5, bins=50)

    plt.subplot(2, 2, 3)
    #plt.title('Patients with diagnosed diabetes', fontsize=10)
    dflm_p['LBXSGL'].plot(kind='kde')
    plt.xlabel('LBXSGL - Glucose (mg/dL)')
    plt.axvline(dflm_p['LBXSGL'].mean(), color='r', linestyle='dashed', linewidth=1);

    plt.subplot(2, 2, 4)
    #plt.title('Patients NOT diagnosed with diabetes', fontsize=10)
    dflm_n['LBXSGL'].plot(kind='kde')
    plt.xlabel('LBXSGL - Glucose (mg/dL)')
    plt.axvline(dflm_n['LBXSGL'].mean(), color='r', linestyle='dashed', linewidth=1);

    plt.tight_layout()
    plt.show()


#Test difference between means, different sample sizes
if False:

    #LBXSOSSI

    df = pd.concat([questionnaire_data['DIAGNOSED_DIABETES'], biochemistry_data['LBXSOSSI']], axis=1)
    dflm_p = df[(df['DIAGNOSED_DIABETES'] == 1)]
    dflm_n = df[(df['DIAGNOSED_DIABETES'] == 0)]


    from scipy.stats import normaltest

    k2, pval = normaltest(dflm_p['LBXSOSSI'].values)
    alpha = 0.05

    if pval < alpha:
        print("The population in {} is not normally distributed".format('LBXSOSSI'))
    else:
        print("The population is normally distributed")


    print("Median of {} in Positive Group".format(dflm_p['LBXSOSSI'].median()))
    print("Median of {} in Negative Group".format(dflm_n['LBXSOSSI'].median()))

    from scipy.stats import kruskal

    H_stat, kpval= kruskal(dflm_p['LBXSOSSI'].values, dflm_n['LBXSOSSI'].values)

    print(H_stat, kpval)


    #LBXSATSI
    if False:

        df = pd.concat([questionnaire_data['DIAGNOSED_DIABETES'], biochemistry_data['LBXSATSI']], axis=1)
        dflm_p = df[(df['DIAGNOSED_DIABETES'] == 1)]
        dflm_n = df[(df['DIAGNOSED_DIABETES'] == 0)]

        k2, pval = normaltest(dflm_p['LBXSATSI'].values)
        alpha = 0.05

        if pval < alpha:
            print("The population in {} is not normally distributed".format('LBXSATSI'))
        else:
            print("The population is normally distributed")






# Histogram and density plots of LBXSOSSI - Osmolality  levels prevalence in patients diagnosed/not-diagnosed with diabetes
if False:
    df = pd.concat([questionnaire_data['DIAGNOSED_DIABETES'], biochemistry_data['LBXSOSSI']], axis=1)
    dflm_p = df[(df['DIAGNOSED_DIABETES'] == 1)]
    dflm_n = df[(df['DIAGNOSED_DIABETES'] == 0)]

    plt.figure(1, figsize=(10, 8))

    # plt.suptitle('Osmolality Level Statistics for Patients with/without Diagnosed Diabetes', fontsize=12)

    plt.subplot(2, 2, 1)
    plt.title('Patients with diagnosed diabetes', fontsize=10)
    dflm_p['LBXSOSSI'].plot(kind='hist', histtype='stepfilled', alpha=0.5, bins=50)
    # plt.xlabel('LBXSOSSI - Osmolality (mmol/Kg)')

    plt.subplot(2, 2, 2)
    plt.title('Patients NOT diagnosed with diabetes', fontsize=10)
    # plt.xlabel('LBXSOSSI - Osmolality (mmol/Kg)')
    dflm_n['LBXSOSSI'].plot(kind='hist', histtype='stepfilled', alpha=0.5, bins=50)

    plt.subplot(2, 2, 3)
    # plt.title('Patients with diagnosed diabetes', fontsize=10)
    dflm_p['LBXSOSSI'].plot(kind='kde')
    plt.xlabel('LBXSOSSI - Osmolality (mmol/Kg)')
    plt.axvline(dflm_p['LBXSOSSI'].mean(), color='r', linestyle='dashed', linewidth=1);

    plt.subplot(2, 2, 4)
    # plt.title('Patients NOT diagnosed with diabetes', fontsize=10)
    dflm_n['LBXSOSSI'].plot(kind='kde')
    plt.xlabel('LBXSOSSI - Osmolality (mmol/Kg)')
    plt.axvline(dflm_n['LBXSOSSI'].mean(), color='r', linestyle='dashed', linewidth=1);

    plt.tight_layout()
    plt.show()





# Test external function defined in gut module
if False:
    gut.hist_density_plots_bc(questionnaire_data,biochemistry_data, 'DIAGNOSED_DIABETES', 'LBXSOSSI', 'Osmolality (n.u.)',
                              mean_mark=False)

#Generate a 2Dimensional histogram using two continuous features
if False:
    plt.hist2d(biochemistry_data['LBXSATSI'], biochemistry_data['LBXSASSI'], bins=60, cmap='Spectral')
    cb = plt.colorbar()
    cb.set_label('counts in bin')
    plt.show()


#Test plotting two scatter plots together
if False:
    plt.figure(1, figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(biochemistry_data['LBXSATSI'].values, biochemistry_data['LBXSASSI'].values, c='orange',
                alpha=0.3, edgecolors='brown')

    plt.subplot(1, 2, 2)
    plt.scatter(biochemistry_data['LBXSATSI'].values, biochemistry_data['LBXSCK'].values, c='green',
                alpha=0.08, edgecolors='yellow')
    plt.show()


#Test plotting scatter and 2dhistogram together
if False:
    plt.figure(1, figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(biochemistry_data['LBXSATSI'].values, biochemistry_data['LBXSASSI'].values, c='orange',
                alpha=0.3, edgecolors='brown')

    plt.subplot(1, 2, 2)
    plt.hist2d(biochemistry_data['LBXSATSI'], biochemistry_data['LBXSASSI'], bins=60, cmap='hot')
    cb = plt.colorbar()
    cb.set_label('counts in bin')
    plt.show()