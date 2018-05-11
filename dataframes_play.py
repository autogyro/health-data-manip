#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:02:43 2017
@author: juanerolon
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys



coef = [0.5, 0.5]

d = {'X1':[2 ,4, 6],'X2':[10, 12, 8]}
df = pd.DataFrame(d, index=['a', 'b', 'c'])


print(df)

if False:
    for m, feat in enumerate(df.columns):
        df[feat] = coef[m] * df[feat]
    print(df)
    df['Y1'] = df['X1'] + df['X2']
    print(df)

def weighted_Sum(df, features, weights, flabel):
    """
    Returns the weighted sum of selected feature in dataframe
    :param df: input dataframe
    :param features: selected features to be weighted
    :param weights: list of weight coefficients
    :param flabel: (str) label name of feature returned by function
    :return:
    """

    frame = df.copy(deep=True)

    for m, feat in enumerate(features):
        frame[feat] = weights[m] * frame[feat]

    frame[flabel] = frame.sum(axis=1)

    return frame[flabel]

print("\nAggregated feature:\n")

f = feat_Comb(df, list(df.columns), coef, 'YX')
print(f)




if False:
    #Test for creating df for storing combinatorics results on feature selection
    #see feature_scan_cv codes

    d = {'FEATURES':[], "ROC":[], "ACC":[]}
    df = pd.DataFrame(d)

    tmp = (['A', 'B'], 80.0, 3.0)

    df = df.append(pd.DataFrame({"FEATURES":[tmp[0]], "ROC":[tmp[1]], "ACC":[tmp[2]]}))

    print(df)




#Combinations itertools testing
if False:
    from itertools import combinations

    #assume "A", "B",... are features; we decide to find all possible combinations of two distinct features
    combs = combinations(iterable=["A", "B", "C", "D"], r=2)

    feat_sets = []
    for el in combs:
        feat_sets.append(list(el))

    print(feat_sets)

    #Combinations itertools testing 2
    l =[]
    for i in range(21):
        l.append(i)
    combs = combinations(iterable=l, r=7)
    combs = list(combs)

    #print(combs)
    print(len(combs))

#print(combs[0], combs[30045010],combs[10045010])

#XGBoost Dmatrix Test
if False:
    import xgboost as xgb

    #create a dataframe
    d = {'X1':[0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
         'X2':[0,0,1,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,1,0,0], 'y':[0,1,0,1,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,1]}

    df = pd.DataFrame(d)

    #convert dataframe to dmatrix data structure
    dtrain = xgb.DMatrix(df)

    #get the features/label names
    print(dtrain.feature_names)
    #get the types of the data in label/feature columns
    print(dtrain.feature_types)



# Oversampling testing
if False:

    from imblearn.over_sampling import SMOTE

    d = {'X1':[0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
         'X2':[0,0,1,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,1,0,0], 'y':[0,1,0,1,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,1]}
    df = pd.DataFrame(d)
    print(df)
    print("")
    X = df[['X1','X2']]
    y = df.y

    print("Original df counts:\n")
    minc = df.y.sum()
    majc = df.y.count()-df.y.sum()
    imbratio = np.round(minc/majc,2)
    print("Minority y class count: {}".format(minc))
    print("Majority y class count: {}".format(majc))
    print("Balancing Ratio: {}".format(imbratio))

    X_resampled, y_resampled = SMOTE(ratio='minority').fit_sample(X, y)

    Xr = pd.DataFrame(X_resampled)
    yr = pd.DataFrame(y_resampled)

    Xdefault_labels = Xr.columns
    Xnew_labels = ['X1', 'X2']

    # Rename dataframe labels
    for n, new_name in enumerate(Xnew_labels):
        Xr.rename(columns={Xdefault_labels[n]: new_name}, inplace=True)

    Ydefault_labels = yr.columns
    Ynew_labels = ['y']

    # Rename dataframe labels
    for n, new_name in enumerate(Ynew_labels):
        yr.rename(columns={Ydefault_labels[n]: new_name}, inplace=True)


    dft = pd.concat([Xr, yr],axis=1)
    print(dft)
    print("")

    for col in Xr.columns:
        Xr[col] = np.round(Xr[col],0)
        Xr[col] = pd.to_numeric(Xr[col], downcast='integer')

    for col in yr.columns:
        yr[col] = np.round(yr[col],0)
        yr[col] = pd.to_numeric(yr[col], downcast='integer')

    dfp = pd.concat([Xr, yr],axis=1)

    print(dfp)
    print("")

    print("Oversampled counts:\n")

    minc = dfp.y.sum()
    majc =dfp.y.count()-dfp.y.sum()
    imbratio = np.round(minc/majc,2)
    print("Minority y class count: {}".format(minc))
    print("Majority y class count: {}".format(majc))
    print("Balancing Ratio: {}".format(imbratio))





def create_binarized_df(input_df, features, thresholds, full_binarization=True):
    """
    :param input_df: dataframe containing the features to be binarized
    :param features: list of features to be binarized
    :param thresholds: threshold values to be passed to sklearn's binarize preprocessing method
    :param full_binarization: boolean value that sets whether we return a version of input_df with all its features
    binarized (True) or a partial dataframe containing the binarized version of the specified
    features in the features list (False).
    :return:
    """
    from sklearn.preprocessing import binarize

    if full_binarization:
        if (len(features) < len(input_df.columns)):
            raise Exception("The list of input features must contain all features in dataframe"
                            "e.g. input_df.columns.")
        if (len(features) > len(input_df.columns)):
            raise Exception("The number of features in features list must be less than or equal to"
                            "the number of all features in input dataframe.")


        frame = input_df.copy(deep=True)
    else:
        if (len(features) > len(input_df.columns)):
            raise Exception("The number of features in features list must be less than or equal to"
                            "the number of all features in input dataframe.")
        frame = input_df[features]

    for feat in features:
        binarize(frame[feat].values.reshape(-1,1), threshold=thresholds[feat], copy=False)
        frame[feat] = pd.to_numeric(frame[feat], downcast='integer')

    return frame


#######################################
## Test create_binarized_df function ##
#######################################

if False:

    d = {'A': [0.1, 0.2, 0.6, 0.3, 0.9, 0.45], 'B': [0.7, 0.8, 0.1, 0.3, 0.9, 0.45], 'C': [0.7, 0.8, 0.1, 0.3, 0.9, 0.45]}
    df = pd.DataFrame(d)

    thr_values = {'A': 0.5, 'B': 0.3, 'C':0.7}

    mydf = create_binarized_df(df,['A','B'],thr_values,full_binarization=False)

    print(mydf)

if False:

    from sklearn.preprocessing import binarize

    d = {'A':[0.1, 0.2, 0.6, 0.3, 0.9, 0.45], 'B':[0.7, 0.8, 0.1, 0.3, 0.9, 0.45]}
    thr_values = {'A':0.5, 'B':0.3}

    df = pd.DataFrame(d)

    print(df)

    df_copy = df.copy(deep=True)

    for feat in df.columns:
        binarize(df[feat].values.reshape(-1,1), threshold=thr_values[feat], copy=False)
        df[feat] = pd.to_numeric(df[feat], downcast='integer')

    print(df)
    print(type(df['A'].values[0]))


if False:
    d = {'A':[0.1, 0.2, 0.6, 0.3, 0.9, 0.45], 'B':[0.7, 0.8, 0.1, 0.3, 0.9, 0.45]}
    df = pd.DataFrame(d)
    df2 = df.copy(deep=True)

    print(df)

    from sklearn.preprocessing import binarize

    binarize(df, threshold=0.5, copy=False)

    df3 = pd.concat([df2['A'], df['A']], axis=1, keys=['AO', 'AB'])

    print(df)
    print(df2)
    print(df3)


#=======================================================================================================================
#
#Computing Risk and Odds Ratios
#===============================


#Mantel–Haenszel estimated relative risk confidence interval:
def confidence_interval(ratio, serror):
    xlci = np.log(ratio) - 1.96*serror
    xuci = np.log(ratio) + 1.96*serror

    lci = round(np.exp(xlci), 2)
    uci = round(np.exp(xuci), 2)

    return (lci, uci)

def relativeRisk(df, disease, exposure, table=False):
    """
    :param df: dataframe containing both disease and exposure (risk factors) records
    :param disease: string: label of the disease feature in dataframe
    :param exposure: string: label of the exposure feature in dataframe
    :param table: boolean: if True use contingency table matrix elements as input; otherwise use disease-exposure df
    :return: dictionary containing relative risk and its confidence interval, plus other info.
    {'Number of Records':nj, 'Number of Exposed Individuals':njp ,
    'Number of Unexposed Individuals':njm, 'Number of Diseased Individuals':DJP,
    'Number of Not Diseased Individuals':DJM, 'Exposed Individuals Risk':RE,
            'Unexposed Individuals Risk':RU, 'Relative Risk':RR, 'Confidence Interval':RR_CI, 'Contingency table':ct}
    """

    if table:
        try:
            a, b, c, d = df[0], df[1], df[2], df[3]
        except:
            raise Exception("Make sure first argument input corresponds to contingency table or tuple")

    if not table:

        try:
            #dataframe containing all records of individuals having the disease
            positve_df = df[df[disease]==1]

            # dataframe containing all records of individuals NOT having the disease
            negative_df = df[df[disease] == 0]

            # Number of positive cases (having disease) exposed to risk
            a = positve_df[positve_df[exposure]==1][exposure].count()

            # Number of negative cases (NOT having disease) exposed to risk
            b = negative_df[negative_df[exposure] == 1][exposure].count()

            # Number of positive cases (having disease) not exposed to risk
            c = positve_df[positve_df[exposure] == 0][exposure].count()

            # Number of negative cases (NOT having disease) not exposed to risk
            d = negative_df[negative_df[exposure] == 0][exposure].count()
        except:
            raise Exception("Make sure first argument input corresponds to disease-exposure dataframe")

    #Backup contingency table ct for output
    s = (2, 2)
    ct = np.zeros(s)
    ct[0, 0], ct[0,1], ct[1,0], ct[1,1] = a, b, c, d


    njp = a + b           #Marginal frequency for exposed group
    njm = c + d           #Marginal frequency for the unexposed group

    nj = njp + njm        #Total number of individuals
    DJP = a + c           #Marginal frequency for the diseased group
    DJM = b + d           #Marginal frequency for the non-diseased group

    _RE = a /njp           #Risk for the exposed group
    RE = np.round(_RE,2)
    _RU = c /njm           #Risk for the unexposed group
    RU = np.round(_RU, 2)

    _RR = _RE / _RU                                              #Relative Risk
    RR = np.round(_RR, 2)
    _RR_SE = np.sqrt(((1/a)+(1/c)) - ((1/(a+b)) + (1/(c+d))))  #Relative Risk Standard Error
    RR_SE = np.round(_RR_SE,2)
    RR_CI = confidence_interval(RR, RR_SE)                    #Relative Risk Confidence Interval

    return {'Records':nj, 'Exposed':njp ,'Unexposed':njm, 'Diseased':DJP, 'NotDiseased':DJM, 'ExposedRisk':RE,
            'UnexposedRisk':RU, 'RelativeRisk':RR, 'RR_CI':RR_CI, 'CTable':ct}


def oddsRatio(df, disease, exposure,table=False):
    """
    :param df: dataframe containing both disease and exposure (risk factors) records
    :param disease: string: label of the disease feature in dataframe
    :param exposure: string: label of the exposure feature in dataframe
    :param table: boolean: if True use contingency table matrix elements as input; otherwise use disease-exposure df
    :return: dictionary containing odds ratio and its confidence interval plus other info
    {'Number of Records':nj, 'Number of Exposed Individuals':njp ,
    'Number of Unexposed Individuals':njm, 'Number of Diseased Individuals':DJP,
    'Number of Not Diseased Individuals':DJM, 'Odds Ratio':OR, 'Confidence Interval':OR_CI, 'Contingency table':ct}
    """

    if table:
        try:
            a, b, c, d = df[0], df[1], df[2], df[3]
        except:
            raise Exception("Make sure first argument input corresponds to contingency table or tuple")

    if not table:

        try:
            #dataframe containing all records of individuals having the disease
            positve_df = df[df[disease]==1]

            # dataframe containing all records of individuals NOT having the disease
            negative_df = df[df[disease] == 0]

            # Number of positive cases (having disease) exposed to risk
            a = positve_df[positve_df[exposure]==1][exposure].count()

            # Number of negative cases (NOT having disease) exposed to risk
            b = negative_df[negative_df[exposure] == 1][exposure].count()

            # Number of positive cases (having disease) not exposed to risk
            c = positve_df[positve_df[exposure] == 0][exposure].count()

            # Number of negative cases (NOT having disease) not exposed to risk
            d = negative_df[negative_df[exposure] == 0][exposure].count()
        except:
            raise Exception("Make sure first argument input corresponds to disease-exposure dataframe")

    #Backup contingency table ct for output
    s = (2, 2)
    ct = np.zeros(s)
    ct[0, 0], ct[0,1], ct[1,0], ct[1,1] = a, b, c, d

    njp = a + b           #Marginal frequency for exposed group
    njm = c + d           #Marginal frequency for the unexposed group

    nj = njp + njm        #Total number of individuals
    DJP = a + c           #Marginal frequency for the diseased group
    DJM = b + d           #Marginal frequency for the non-diseased group

    _OR = (a * d) / (b * c)                                  # Odds Ratio
    OR = np.round(_OR, 2)
    _OR_SE = np.sqrt((1 / a) + (1 / b) + (1 / c) + (1 / d))  # Odds Ratio Standard Error
    OR_SE = np.round(_OR_SE,2)
    OR_CI = confidence_interval(OR, OR_SE)                  # Odds Ratio Confidence Interval

    return {'Records':nj, 'Exposed':njp ,'Unexposed':njm, 'Diseased':DJP, 'NotDiseased':DJM,
            'OddsRatio':OR, 'OR_CI':OR_CI, 'CTable':ct}


def attributableRisk(df, disease, exposure, table=False):
    """
    :param df: dataframe containing both disease and exposure (risk factors) records
    :param disease: string: label of the disease feature in dataframe
    :param exposure: string: label of the exposure feature in dataframe
    :param table: boolean: if True use contingency table matrix elements as input; otherwise use disease-exposure df
    :return: dictionary containing attributable risks and their confidence intervals, plus other info
    {'Number of Records':nj, 'Number of Exposed Individuals':njp ,
    'Number of Unexposed Individuals':njm, 'Number of Diseased Individuals':DJP,
    'Number of Not Diseased Individuals':DJM, {'Records':nj, 'Exposed':njp ,'Unexposed':njm, 'Diseased':DJP, 'NotDiseased':DJM,
            'AttribRisk':np.round(ar,2), 'AR_CI':np.round(ar_ci,2), 'AttribRiskPct':np.round(arp,2), 'ARP_CI':np.round(arp_ci,2),
            'PopultationAR':np.round(par,2), 'PARP_CI':np.round(parp,2), 'Number needed to treat': nnt, 'Contingency table':ct}
    """

    if table:
        try:
            a, b, c, d = df[0], df[1], df[2], df[3]
        except:
            raise Exception("Make sure first argument input corresponds to contingency table or tuple")

    if not table:

        try:
            #dataframe containing all records of individuals having the disease
            positve_df = df[df[disease]==1]

            # dataframe containing all records of individuals NOT having the disease
            negative_df = df[df[disease] == 0]

            # Number of positive cases (having disease) exposed to risk
            a = positve_df[positve_df[exposure]==1][exposure].count()

            # Number of negative cases (NOT having disease) exposed to risk
            b = negative_df[negative_df[exposure] == 1][exposure].count()

            # Number of positive cases (having disease) not exposed to risk
            c = positve_df[positve_df[exposure] == 0][exposure].count()

            # Number of negative cases (NOT having disease) not exposed to risk
            d = negative_df[negative_df[exposure] == 0][exposure].count()
        except:
            raise Exception("Make sure first argument input corresponds to disease-exposure dataframe")

    #Backup contingency table ct for output
    s = (2, 2)
    ct = np.zeros(s)
    ct[0, 0], ct[0,1], ct[1,0], ct[1,1] = a, b, c, d

    njp = a + b           #Marginal frequency for exposed group
    njm = c + d           #Marginal frequency for the unexposed group

    nj = njp + njm        #Total number of individuals
    DJP = a + c           #Marginal frequency for the diseased group
    DJM = b + d           #Marginal frequency for the non-diseased group


    _RE = a /njp          #Risk for the exposed group
    _RU = c /njm          #Risk for the unexposed group
    _RR = _RE / _RU       #Relative Risk

    N = njm

    ar = (a/(a+b))-(c/(c+d))                                           #Attributable Risk
    nnt = 1.0 / ar                                                     #Number of individuals needed to treat
    ar_se = np.sqrt(((a+c)/N)*(1-((a+c)/N))*((1/(a+b))+(1/(c+d))))     #Attributable Risk Standard Error
    ar_ci = (round(ar-(1.96*ar_se), 2), round(ar+(1.96*ar_se), 2))     #Attributable Risk Confidence Interval


    arp = 100*((_RR-1)/(_RR))                                   #Attributable risk percentage
    arp_se = (1.96*ar_se)/ar                                    #Attributable risk percentage standard error
    arp_ci = (round(arp-arp_se, 2), round(arp+arp_se, 3))       #Attributable risk percentage confidence interval

    par = ((a+c)/N) - (c/(c+d))     #Population attributable risk
    parp = 100*(par/(((a+c)/N)))    #Population attributable risk percent



    return {'Records':nj, 'Exposed':njp ,'Unexposed':njm, 'Diseased':DJP, 'NotDiseased':DJM,
            'AttRisk':np.round(ar,2), 'AR_CI':np.round(ar_ci,2), 'AttRiskPct':np.round(arp,2), 'ARP_CI':np.round(arp_ci,2),
            'PopultationAR':np.round(par,2), 'PARP_CI':np.round(parp,2), 'NNT':np.ceil(nnt),'CTable':ct}




if False:

    if False:
        d = {'disease': [1, 1, 0, 0, 1, 0, 1], 'exposure': [1, 0, 0, 1, 1, 1, 0]}
        health_df = pd.DataFrame(d)

    if True:
        health_df = [350, 1200, 200, 1900]

    print("RRD:\n")
    RRD = relativeRisk(health_df,'disease', 'exposure',table=True)
    print(RRD['RelativeRisk'], RRD['RR_CI'])

    print("ORD:\n")
    ORD = oddsRatio(health_df,'disease', 'exposure',table=True)
    print(ORD['OddsRatio'], ORD['OR_CI'])


    print("ATR:\n")
    ATR = attributableRisk(health_df,'disease', 'exposure',table=True)
    print(ATR['AttRisk'], ATR['AttRiskPct'], ATR['ARP_CI'], ATR['NNT'])




print("")


#--------------------------------------------------------------


if False:
    d = {'disease':[1,1,0,0,1,0,1], 'exposure':[1,0,0,1,1,1,0]}
    health_df = pd.DataFrame(d)

    print(health_df)

    #dataframe containing all records of individuals having the disease
    positve_df = health_df[health_df['disease']==1]

    # dataframe containing all records of individuals NOT having the disease
    negative_df = health_df[health_df['disease'] == 0]

    # Number of positive cases (having disease) exposed to risk
    a = positve_df[positve_df['exposure']==1]['exposure'].count()

    # Number of negative cases (NOT having disease) exposed to risk
    b = negative_df[negative_df['exposure'] == 1]['exposure'].count()

    # Number of positive cases (having disease) not exposed to risk
    c = positve_df[positve_df['exposure'] == 0]['exposure'].count()

    # Number of negative cases (NOT having disease) not exposed to risk
    d = negative_df[negative_df['exposure'] == 0]['exposure'].count()

    #Test with data on notebook. This test confirms the correctness of the formulae set below
    #see http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Confidence_Intervals/BS704_Confidence_Intervals8.html
    if False:
        a = 9
        b = 41
        c = 20
        d = 29

    njp = a + b           #Marginal frequency for exposed group
    njm = c + d           #Marginal frequency for the unexposed group

    nj = njp + njm        #Total number of individuals
    DJP = a + c           #Marginal frequency for the diseased group
    DJM = b + d           #Marginal frequency for the non-diseased group

    RE = a /njp           #Risk for the exposed group
    RU = c /njm           #Risk for the unexposed group

    RR = RE / RU                                              #Relative Risk
    RR = np.round(RR, 2)
    RR_SE = np.sqrt(((1/a)+(1/c)) - ((1/(a+b)) + (1/(c+d))))  #Relative Risk Standard Error
    RR_CI = confidence_interval(RR, RR_SE)                    #Relative Risk Confidence Interval

    AR = (RR-1.0)/RR                                          #Attributable Risk [0,1] (multiply by 100 to get pct)
    ARP = 100.0*(RR - 1.0) / RR  # Attributable Risk [0,1] (multiply by 100 to get pct)
    NNT = 1.0/AR                                              #Number Needed to Treat

    OR = (a*d)/(b*c)                            #Odds Ratio
    OR = np.round(OR, 2)
    OR_SE = np.sqrt((1/a)+(1/b)+(1/c)+(1/d))    #Odds Ratio Standard Error
    OR_CI = confidence_interval(OR, OR_SE)      #Odds Ratio Confidence Interval

    print("RR = {}, RR_CI = {}".format(RR, RR_CI))
    print("OR = {}, OR_CI = {}".format(OR, OR_CI))




if False:
    d = {'disease':[1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0], 'indicator':[0.12,1.2,0.54,0.05,1.03,0.43,1.15,0.1,0.2,0.8,0.5,0.2,0.1,1.1,0.23,0.7]}
    health_df = pd.DataFrame(d)

    print("Health Data")
    print(health_df)

    print("Positive Cases")
    df1_p = health_df[(health_df['disease'] == 1)]
    print(df1_p)

    print("Positive below normal")
    df2_p = df1_p[(df1_p['indicator'] < 0.3)]
    print(df2_p)

    a_below = df2_p['indicator'].count()


    print("Positive above normal")
    df3_p = df1_p[(df1_p['indicator'] > 0.6)]
    print(df3_p)

    a_above = df3_p['indicator'].count()

    print("Normal Positives")
    normal_p = df1_p[(df1_p['indicator'] >= 0.3)]
    normal_p = normal_p[normal_p['indicator'] <= 0.6]
    print(normal_p)

    c = normal_p['indicator'].count()
    if c==0:
        c=1

    print("Negative Cases")
    df1_n = health_df[(health_df['disease'] == 0)]
    print(df1_n)

    print("Negative below normal")
    df2_n = df1_n[(df1_n['indicator'] < 0.3)]
    print(df2_n)

    b_below = df2_n['indicator'].count()

    print("Negative above normal")
    df3_n = df1_n[(df1_n['indicator'] > 0.6)]
    print(df3_n)

    b_above = df3_n['indicator'].count()

    print("Normal Negatives")
    normal_n = df1_n[(df1_n['indicator'] >= 0.3)]
    normal_n = normal_n[normal_n['indicator'] <= 0.6]
    print(normal_n)

    d =  normal_n['indicator'].count()

    print("Contingency Table Elements Above Threshold\n")

    print("a = {}".format(a_above))
    print("b = {}".format(b_above))
    print("c = {}".format(c))
    print("d = {}".format(d))
    RR = (a_above/(a_above+b_above))/(c/(c+d))
    print("Risk Ratio = {}".format(RR))

    print("Contingency Table Elements Below Threshold\n")

    print("a = {}".format(a_below))
    print("b = {}".format(b_below))
    print("c = {}".format(c))
    print("d = {}".format(d))
    RR = (a_below/(a_below+b_below))/(c/(c+d))
    print("Risk Ratio = {}".format(RR))

    print("Contingency Table Elements Outside Normal Range\n")

    print("a = {}".format(a_above + a_below))
    print("b = {}".format(b_above + b_below))
    print("c = {}".format(c))
    print("d = {}".format(d))
    a = a_above + a_below
    b = b_above + b_below
    RR = (a/(a+b))/(c/(c+d))
    print("Risk Ratio = {}".format(RR))



#the purpose is to develo risk ratio for disease when levels are below normal and above normal,
#i.e. when levels are out of normal range
#function prototype to compute relative risk (for developing disease) ratio

def get_risk_ratios(input_df, disease_feature, level_feature, below_normal_level, above_normal_level):
    postive_cases = input_df[disease_feature == 1]
    upper_positive_df = postive_cases[(positive_cases[level_feature] >= above_normal_level)]

    print(upper_positive_df.describe())


#get_risk_ratios(health_df, 'disease', 'indicator', 0.3, 0.6)

#Build a symmetric matrix from linear array
if False:
    cols = ['a','b','c']
    L = len(cols) * len(cols) * [0.0]
    print(L)
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            m = i * len(cols) + j
            n = j * len(cols) + i
            L[m] = cols[i] + cols[j]
            L[n] = L[m]

    BB = np.reshape(L, (-1, len(cols)))
    print(BB)


#Test droping records corresponding to specified indexes
if False:
    d = {'x': [0.1, 0.2, 0.15, 0.25], 'y': [0.1, 200.0, 0.15, 0.25], 'seqn':[300,301,302,303]}
    df = pd.DataFrame(d)
    print(df)
    print("")
    df.set_index('seqn', inplace=True)
    print(df)
    print("")
    df.drop([301, 302], inplace=True)
    print(df)
    print("\nNo. Keys in {}: {}\n".format(df.keys(), len(df.keys())))


#Test filtering outliers the hard way
if False:
    from importlib.machinery import SourceFileLoader
    mod_path = "/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/nhanes2013-2104/"
    txt = SourceFileLoader("text_utils", mod_path+"text_utils.py").load_module()

    d = {'x':[0.1, 0.2, 0.15, 0.25, 3.0, 5.0, 100.0], 'y':[0.1, 200.0, 0.15, 0.25, 3.0, 5.0, 0.1]}
    df = pd.DataFrame(d)
    print(df)

    rf = txt.filter_outliers(df)

    print(rf)



if False:
    from sklearn.metrics import confusion_matrix

    d = {'y_true':[1,0,1,0,0,0,1,1], 'y_pred':[1,0,1,1,0,1,1,0]}
    df = pd.DataFrame(d)

    print(df)
    CM = confusion_matrix(df['y_true'], df['y_pred'])
    print("Confusion Matrix:\n {} \n".format(CM))
    print(np.array([['TN', 'FP'], ['FN', 'TP']]))



if False:
    d = {'a':[0.1, 1.5, 8.0, 0.01, 10.0, 3.5], 'b':[0.1, 1.5, 8.0, 0.01, 10.0, 3.5], 'c':[0.1, 1.5, 8.0, 0.01, 10.0, 3.5]}
    df = pd.DataFrame(d)
    print(df)
    skewed = ['a', 'c']
    df[skewed] = df[skewed].apply(lambda x: np.log(x + 1))
    print(df)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler() # default=(0, 1)
    numerical_features = ['a', 'b', 'c']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    print(df)

if False:

    d = {'a':[np.nan,2,3, np.nan, 4, 5], 'b': [np.nan, np.nan, 5,6,7,8]}
    df = pd.DataFrame(d)
    print(df)

    dfn = df[df.isnull().any(axis=1)]
    tot = len(dfn.index)
    print(dfn)
    print("Num of record rows with nans: {}".format(tot))


#Test set operations on dataframes

if False:
    a = np.array([1, 2, 3, 2, 4, 1])
    b = np.array([3, 4, 5, 6])

    difab = np.setdiff1d(a, b)
    difba = np.setdiff1d(b, a)
    interab = np.intersect1d(a, b)

    print(difab)
    print(difba)
    print(interab)

    d = {'a': [1, 2, 3, 2, 4, 1], 'b': [3, 4, 5, 6]}

    df = pd.DataFrame.from_dict(d, orient='index')
    df = df.transpose()
    # df.fillna(value=0, inplace=True)
    print(df)

    difab = np.setdiff1d(df.a, df.b)
    print(difab)

    difba = np.setdiff1d(df.b, df.a)
    print(len(difba))



# Test logic gate operations across column elements
if False:

    d={'input_1':[0,0,1,1], 'input_2':[0,1,0,1]}
    df= pd.DataFrame(d)


    def bit_logic(x, y, op):
        """Implements bitwise boolean operations among bits x and y
        Bit states are represented by integers 0,1"""
        if op == 'OR':
            return int((x or y))
        elif op =='AND':
            return int((x and y))
        elif op == 'NAND':
            return int(not (x and y))
        elif op == 'NOR':
            return int(not (x or y))
        elif op == 'XOR':
            return int((x and (not y)) or ((not x) and y))
        else:
            raise Exception("Incorrect Boolean operator selection")

    print("OR GATE")
    df['output'] = np.vectorize(bit_logic)(df['input_1'].values, df['input_2'].values, 'OR')
    print(df)
    print("XOR GATE")
    df['output'] = np.vectorize(bit_logic)(df['input_1'].values, df['input_2'].values, 'XOR')
    print(df)

if False:

    # Random sample of reals in [a,b] (b - a) * random_sample() + a
    #
    # Three-by-two array of random numbers from [-5, 0)
    # 5 * np.random.random_sample((3, 2)) - 5
    #

    # Num elements in sample
    num_elements = 10

    # Interval where random numbers live
    a, b = 1.0, 10.0

    # Float values random generator
    data = a + (b - a) * np.random.random_sample(num_elements)
    # Binary values random generator
    data2 = np.random.randint(2, size=num_elements)
    # Bin size
    bin_size = 2.0

    # Built generic dataframe for data above
    d = {'sample': data, 'value': data2}
    df = pd.DataFrame(d)


#Test global value replacement
if False:
    dx = {'a':[1,2,2,1], 'b':[2,2,0,0]}
    dxf = pd.DataFrame(dx)
    print(dxf)
    print("")
    dxf.replace(to_replace=2, value='two', inplace=True)

    print(dxf)



#Test replacing values in dataframe series
if False:
    print(df)

    df.value.replace(to_replace=1, value=2, inplace=True, limit=None, regex=False, method='pad', axis=None)

    print(df)

def restrict_by_interval(df, feature, min_val, max_val, boundary):
    """Restricts daframe to records containing values inside the interval specified
    for a continuous feature.
    Inputs:
    dataframe: df
    continuous feature: feature
    lower limit: min_val
    upper limit: max_val
    boundary or interval type: boundary: 'inclusive', 'exclusive', 'left', 'right'
    """

    if (boundary == 'inclusive'):
        dflm = df[(df[feature] >= min_val)]
        dflm = dflm[dflm[feature] <= max_val]
        return dflm
    elif (boundary == 'exclusive'):
        dflm = df[(df[feature] >= min_val)]
        dflm = dflm[dflm[feature] <= max_val]
        return dflm
    elif (boundary == 'left'):
        dflm = df[(df[feature] >= min_val)]
        dflm = dflm[dflm[feature] < max_val]
        return dflm
    elif (boundary == 'right'):
        dflm = df[(df[feature] > min_val)]
        dflm = dflm[dflm[feature] <= max_val]
        return dflm
    else:
        raise Exception("Incorrect interval boundary specificiation.\n"
                        " Choose between 'inclusive []', 'exclusive ()', 'left [)', 'right (]'")
        return None

if False:
    print(df.head(11))

    df2 = restrict_by_interval(df,'sample', 2,6,'righ')

    print(df2)


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

#print(gcd(71,20))

def plot_binary_histogram(df, feature, title, xlabel, ylabel,
                          ytrue_label, yfalse_label):
    """Creates a bar plot of a single feature that takes binary values (0, 1).
    input: dataframe(df), feature string name (feature)"""

    s1 = df[df[feature] == 1][feature].count()
    s0 = df[df[feature] == 0][feature].count()
    bars = [s1, s0]
    s1_array = [s1]
    s0_array = [s0]

    maxval = np.max(bars)

    index = np.arange(len([1]))
    bar_width = 0.1
    opacity = 0.8

    plt.bar(index, s0_array, bar_width, alpha=opacity, color='b', label=yfalse_label)
    plt.bar(index + bar_width, s1_array, bar_width, alpha=opacity, color='r', label=ytrue_label)

    plt.ylim(0, maxval*1.25)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.xticks(index, bar_labels, rotation='horizontal')
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')  # labels along the bottom edge are off
    plt.legend(frameon=False, loc='upper right', fontsize='small')


if False:

    plot_binary_histogram(df, 'value', 'title', 'val label', 'ylabel',
                              'ytrue', 'yfalse')
    plt.show()


def twofeat_barplot(df, xfeature, yfeature, nbins, title,
                    xlabel, ylabel, ytrue_label, yfalse_label,xticks_rotation='horizontal', verbose=False):

    """Plots a barplot histogram-like of the number of records corresponding to True and False values of a
    dataframe column 'yfeature' versus a continuous-valued column 'xfeature'. The x-axis bins are computed
    automatically after specifying the number of bins requested at input. The chart contains two bars per
    bin, corresponding each to a boolean True (0) or False (1).

    inputs:

    df: dataframe
    xfeature: string name of the column specified as x-axis in the plot
    yfeature: string name of the column speficied ax y-axis in the plot. yfeature values are binary (1 or 0)
    nbins: number of bins for the barplot.
    title: chart title
    xlabel: x-axis label
    ylabel: y-axis label
    ytrue_label: label for bar assigned to number of True values in yfeature column
    yfalse_label: label for bar assigned to number of False values in yfeature column
    xticks_rotation: orientation of bin labels as specified by matplotlib
    verbose: if True prints summary counts info for each dataframe used to plot bars generated per bin

    """

    max = np.max(df[xfeature].values)
    min = np.min(df[xfeature].values)
    rng = np.abs(max-min)
    bin_size = np.round(rng/nbins,0)

    #Create bins
    bins = []
    bins_str = []

    span = np.floor(min)

    while span <= np.ceil(max) + bin_size:
        bins.append(span)
        span += bin_size

    bins = np.round(bins, 2)

    for m in range(len(bins) -1 ):
        label = "{} to {}".format(bins[m], bins[m+1])
        bins_str.append(label)

    #Fill bars over bins
    s1_array = []
    s0_array = []

    for m in range(len(bins)-1):
        df1 = df[(df[xfeature] >= bins[m])]
        df2 = df1[df1[xfeature] < bins[m+1]]
        nv =  df2[xfeature].count()

        s1 = df2[df2[yfeature] == 1][yfeature].count()
        s0 = df2[df2[yfeature] == 0][yfeature].count()
        s1_array.append(s1)
        s0_array.append(s0)

        if verbose:
            print("Dataframe Bin")
            print(df2)
            print("Bin: " + bins_str[m])
            print("Sample count: {}".format(nv))
            print("")
            print("True counts s1 = {}, False counts s0={}".format(s1,s0))
            print("")


    index = np.arange(len(bins_str))
    bar_width = 0.35
    opacity = 0.8

    plt.bar(index, s0_array, bar_width, alpha=opacity, color='k', label=yfalse_label)
    plt.bar(index + bar_width, s1_array, bar_width, alpha=opacity, color='g', label=ytrue_label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(index + bar_width / 2.0, bins_str, rotation=xticks_rotation)
    plt.legend(frameon=False, loc='upper right', fontsize='small')



if False:
    title = 'Generic Bar Plot'
    xlabel = 'Random Numbers'
    ylabel = 'No. Records'
    ytrue_label = '1'
    yfalse_label = '0'

    twofeat_barplot(df, 'sample', 'value', 5, title, xlabel, ylabel, ytrue_label, yfalse_label)
    plt.tight_layout()
    plt.show()

if False:

    #Random sample of reals in [a,b] (b - a) * random_sample() + a
    #
    #Three-by-two array of random numbers from [-5, 0)
    # 5 * np.random.random_sample((3, 2)) - 5
    #

    #Num elements in sample
    num_elements = 100

    #Interval where random numbers live
    a, b = 1.0, 10.0


    #Float values random generator
    data = a + (b-a)*np.random.random_sample(num_elements)
    #Binary values random generator
    data2 = np.random.randint(2, size=num_elements)
    #Bin size
    bin_size = 2.0

    #Built generic dataframe for data above
    d = {'sample': data, 'value': data2}
    df = pd.DataFrame(d)

    #Simple stats
    max = np.round(np.max(df['sample'].values),4)
    min = np.round(np.min(df['sample'].values),4)
    rango = np.round((max-min),4)

    print("Dataframe head:\n")
    print(df.head(10))
    print("")
    print("max: {}, min: {}, range: {}\n".format(max, min, rango))
    print("Ceiling max: {}".format(np.ceil(max)))
    print("Floored min: {}".format(np.floor(min)))

    #Create bins
    bins = []
    span = np.floor(min)

    while span <= np.ceil(max) + bin_size:
        bins.append(span)
        span += bin_size

    bins = np.round(bins, 2)
    print(bins)
    print("")
    #----------------------------------------------

    #Create bin's labels
    bins_str = []
    for m in range(len(bins) -1 ):
        label = "{} to {}".format(bins[m], bins[m+1])
        bins_str.append(label)
    print(bins_str)
    print("")
    print("-----------------------------------------")


    #Fill appropiate continuous feat bins and corresponding count of feat2 binary values
    verbose = True
    feat = 'sample'
    feat2 = 'value'
    s1_array = []
    s0_array = []
    for m in range(len(bins)-1):
        df1 = df[(df[feat] >= bins[m])]
        df2 = df1[df1[feat] < bins[m+1]]
        nv =  df2[feat].count()
        s1 = df2[df2[feat2] == 1][feat2].count()
        s0 = df2[df2[feat2] == 0][feat2].count()
        s1_array.append(s1)
        s0_array.append(s0)

        if verbose:
            print("Dataframe Bin")
            print(df2)
            print("Bin: " + bins_str[m])
            print("Sample count: {}".format(nv))
            print("")
            print("True counts s1 = {}, False counts s0={}".format(s1,s0))
            print("")


    index = np.arange(len(bins_str))
    bar_width = 0.35
    opacity = 0.8

    plt.bar(index, s0_array, bar_width, alpha=opacity, color='k', label='0 values')
    plt.bar(index + bar_width, s1_array, bar_width, alpha=opacity, color='g', label='1 values')

    plt.xlabel('Bins')
    plt.ylabel('Binary Counts')
    plt.title('Generic plot')
    plt.xticks(index + bar_width / 2.0, bins_str)
    plt.legend(frameon=False, loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.show()

if False:

    d = {'A': [1, 2, 2, 3, 4, 4, 5, 5], 'B': [1.1, 2.3, 2, 3, 4, 4.3, 5, 5.1], 'C': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']}
    df = pd.DataFrame(d)

    print(df[['A', 'C']])


if False:
    # Extract unique elements from dataframe column (or series)
    d = {'A': [1, 2, 2, 3, 4, 4, 5, 5], 'B': [1.1, 2.3, 2, 3, 4, 4.3, 5, 5.1]}
    df = pd.DataFrame(d)
    print(df)
    print(df.A.unique())

if False:

    def percent_transform(x,n):
        return np.round(float(x/n *100.0),2)

    print(percent_transform(7,21))


if False:
    print("\nMinMaxScaler:\n")
    scaler = MinMaxScaler()
    scaler.fit(df)
    nptdf = scaler.transform(df)
    nptdf = np.round(nptdf ,2)
    print(pd.DataFrame(nptdf,columns=['A', 'B']))

if False:
    print("\nNormalizer:\n")
    scaler = Normalizer()
    scaler.fit(df)
    nptdf = scaler.transform(df)
    nptdf = np.round(nptdf ,2)
    print(pd.DataFrame(nptdf,columns=['A', 'B']))




if False:
    #Illustrates you can apply numPy functions to entire dataframes
    d = {'a':[1.11331, 9.53626], 'b':[0.09144, 53.44221]}
    df = pd.DataFrame(d)
    print(df)
    print(np.round(df,2))


if False:

    d = {'a': [1, 2, 3], 'b':[4, 5, 6]}
    df = pd.DataFrame(d)
    print(df)

    def foo2D(x, y):
        return x + y

    df['c'] = foo2D(df['a'], df['b'])
    print(df)

if False:
    d = {'a':[1,2,3]}
    df = pd.DataFrame(d)

    print(df)

    def foo(x):
        return x+1

    df['b'] = foo(df['a'])

    print(df)



if False:
    #In this example we use the pandas apply method and a pair of functions
    #to re-encode the values of a dataframe column according to a given
    #criterion
    def foo(x):
        if (x==1) or (x==2):
            return 'Yes'
        else:
            return np.nan

    d = {'a':[1,2,1,1,1,2,3,2,1,2,2,3], 'b':[1.0,5.0,1.0,1.0,2.0,4.0,3.0,8.0,1.0,3.0,2.0,3.0]}
    df = pd.DataFrame(d)
    df['c'] = df['a'].apply(lambda x: foo(x))

    #This statement will be useful to convert an il-defined float index to int;
    #Implement policy of assuring integer valued indexes in all dataframes
    df['d'] = pd.to_numeric(df['b'], downcast='integer')

    print(df)

if False:
    # Clear rows containing nans; IN-PLACE; method modifies supplied dataframe, returns None
    d = {'A': [1.1, 2.5, np.nan, 4.8, 5.1, 6.3, np.nan], 'B': [np.nan, 4.0, 3.3, np.nan, 4.0, 5.0, 6.0]}
    df = pd.DataFrame(d)
    print(df)

    # Remove rows containing NaNs in any column field
    print("\nDrop rows with NaNs:\n")
    df.dropna(axis=0, how='any', inplace=True)
    print(df)


    #Clear rows based on specific value found for a particular column field

    print("\nDrop rows such that the B column contains a 4.0:\n")
    nfs = df[df.B == 4.0].B.count()

    df = df[df.B != 4.0]

    print(df)

    print("Number of rows deleted: {}".format(nfs))

if False:
    #Clear rows containing nans; not inplace; method returns new dataframe object
    d = {'A':[1.1, 2.5, np.nan, 4.8, 5.1, 6.3, np.nan], 'B':[np.nan, 3.2, 3.3, np.nan, 4.0, 5.0,6.0]}
    df = pd.DataFrame(d)
    print(df)

    #Remove rows containing NaNs in any column field of df; returns to df1
    df1 = df.dropna(axis=0, how='any')
    print("\nDrop rows with NaNs:\n")
    print(df1)




if False:
    #This example helps mapping a single number value to a single label
    #using bining; will help in relabeling numeric values to labels such
    #as income codes to labels specifying ranges;

    d = {'A':[1, 2, 3, 4, 5, 12, 77, 99]}

    df = pd.DataFrame(d)

    print(df)

    bins = [0,1,2,3,4,5,12,77,99]

    print(len(bins))

    cats = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']

    df['bins'] = pd.cut(df['A'], bins)
    categories = pd.cut(df['A'], bins, labels=cats)

    df['Labels'] = categories

    print(df)

if False:

    XP = np.round(np.linspace(0.0, 2.0*np.pi, num=20),1)
    YP = np.round(np.cos(XP),2)
    ZP = np.round(np.sin(XP),2)

    d = {'XP':XP, 'YP':YP, 'ZP':ZP}
    df = pd.DataFrame(d)

    print("\nData frame\n:")
    print(df)

    Xbins = np.round(np.linspace(0.0, 2.0*np.pi, num=5),1)
    group_names = ['Bad', 'Okay', 'Good', 'Great']

    print("\nBins:\n")
    print(Xbins)

    df['Xbins'] = pd.cut(df['XP'], Xbins)
    categories = pd.cut(df['XP'], Xbins, labels=group_names)

    df['Level'] = categories


    print("\nData frame including binned data column for XP values \n:")
    print(df)


    ax = df[['YP','ZP']].plot(kind='bar', title ="Bar plot", figsize=(10, 5), legend=True, fontsize=12)
    ax.set_xlabel("Index", fontsize=12)
    ax.set_ylabel("YP, ZP", fontsize=12)
    plt.show()




if False:
    df[['YP', 'ZP']].hist(by=df['Level'], bins=9, facecolor=['green'], alpha=0.75, linewidth=1,
                                   edgecolor='black')
    plt.show()


#Access record by specifying indexes
if False:

    d = {'A':[1.0,2.5,3.0], 'B':[1.1, 2.2, 3.3], 'C':[3.5, 4.0, 1.0], 'D':[1,2,3]}
    df = pd.DataFrame(d)
    print(df)

    df2 = pd.concat([df['A'], df['C'], df['D']], axis=1, keys=['A', 'C', 'D'])

    print(" ")
    print(df2)

    df3 = df2.set_index('D')

    print(" ")
    print(df3)

    print(" ")
    print(df3.loc[2])

    print(" ")
    print(df3.loc[2]['C'])

    df4 = df.copy(deep=True)

    print(" ")
    print(df4)

    df4 = df4.drop(df4.index[[0,2]])

    print(" ")
    print(df4)


if False:
    d1 = {'A':[1.0, 3.1, 2.0, 5.5], 'B':[0.1, 1.1, 2.4, 10.5]}
    d2 = {'A2': [1.0, 3.1, 2.0], 'B2': [0.1, 2.4, 10.5]}
    print("df1:\n")
    df1 = pd.DataFrame(d1)
    df2 = pd.DataFrame(d2, index=[2,3,4])

    print(df1)
    print("")
    print(df2)

    print("\ndf1 reindexed with df2 indexes:\n")
    df1 = df1.reindex(list(df2.index))
    print(df1)

if False:
    x = ['A']*300 + ['B']*400 + ['C']*300
    y = np.random.randn(1000)
    df = pd.DataFrame({'Letter':x, 'N':y})


    df['N'].hist(by=df['Letter'])
    plt.show()


if False:

    s1 = pd.Series(['a', 'b', 'c', 'd'])
    s2 = pd.Series(['x1', 'x2', 'x3', 'x4', 'x5'], index=[2,3,41,51,61])

    print("Indexex from series s1 and s2:\n")
    print(list(s1.index))
    print(list(s2.index))

    print("Reindex of series s2 using the indexes of series s1:\n")
    s2r  = s2.reindex(list(s1.index))

    print(s2r)


if False:
    print("\n----------------------------------\n")

    print("Series alignment: first tuple element")
    print(s1.align(s2)[0])
    print("Series alignment: second tuple element")
    print(s1.align(s2)[1])



#Extracting series data from dataframes
if False:
    d = {'A':[1,2,3,4], 'B':[5,6,7,8], 'C':[9,10,11,12]}
    df = pd.DataFrame(data=d)

    df.plot.scatter(x='A', y='B')
    plt.show()

    if False:

        print(df)

        print("Defining a new series\n")
        new = pd.Series(list(df.C), index=list(df.A))
        print(new)
        print("Get a single element index=3:\n")
        print(new[3])




def create_canvas(axlabels):
    # frame
    
    plt.axes()
    plt.xlim(-1.2, 1.2)
    plt.ylim(-0.8, 1.8)    
    plt.axis(axlabels)


def plotx(mystr):
    x = 'Voltron'
    return plt.text(0.5,0.5,mystr+x)


if False:

    plt.figure(1, figsize=(10,5)) 
    
    plt.subplot(1,2,1)
    
    plt.plot()
    plt.xlim(-1.4, 1.4)
    plt.ylim(-0.8, 1.8)    
    plt.axis('on')   
    plt.xticks([])
    plt.yticks([])
    
    
    plt.subplot(1,2,2)
    
    
    plt.plot()
    plt.xlim(-1.2, 1.2)
    plt.ylim(-0.8, 1.8)    
    plt.axis('on')
    xi = -0.8
    yi = 0.5
    dx = 0.4
    dy = 0.0
    plt.arrow(xi, yi, dx,dy, head_width=0.04, head_length=0.05, fc='b', ec='b')
    plt.text(-0.8,0.6,'Oncoming left', rotation='horizontal', fontsize='9')


#Test1
if False:

    plt.subplot(1,2,1)
    plotx('Mega')
    
    plt.subplot(1,2,2)
    plotx('Super')


#---------- CNN Multiclass classification example --------------------------------------------------
#https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

if False:

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.utils import np_utils
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import Pipeline

    dataset_path = '/Users/juanerolon/Desktop/'
    df = pd.read_csv(dataset_path + "iris.csv", header=None)
    dataset = df.values
    X = dataset[:,0:4].astype(float)
    Y = dataset[:,4]

    if False:
        print(df.columns)
        print(df.head())
        print("\nDataset:\n")
        print(dataset)
        print("\nY:\n")
        print(Y)
        print("\nX:\n")
        print(X)

    # encode column containing classes values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)# encode class values as integers

    if False:
        print(encoded_Y)
        print("")
        print(X)
        print(dummy_y.shape[1])

    if False:
        #my version (shorter...)
        dummy_df = pd.get_dummies(df[4])
        my_dummy_y = dummy_df.values

        if False:
            print(dummy_df.head())
            print(my_dummy_y)

    """There is a KerasClassifier class in Keras that can be used as an Estimator in scikit-learn,
    the base type of model in the library. The KerasClassifier takes the name of a function as an argument.
    This function must return the constructed neural network model, ready for training."""

    """Below is a function that will create a baseline neural network for the iris classification problem.
    It creates a simple fully connected network with one hidden layer that contains 8 neurons.
    The hidden layer uses a rectifier activation function which is a good practice.
    Because we used a one-hot encoding for our iris dataset, the output layer must create 3 output values,
    one for each class. The output value with the largest value will be taken as the class predicted by the model."""

    num_features = X.shape[1]       # Get te number of columns features matrix
    num_classes = dummy_y.shape[1]  # Get te number of columns in one-hot encoded matrix of classes

    # Define baseline model

    def baseline_model():
        # Create model
        model = Sequential()
        num_neurons_hidden_1 = 8
        model.add(Dense(num_neurons_hidden_1, input_dim=num_features, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], verbose=1)
        return model

    estimator = KerasClassifier(build_fn=baseline_model, epochs=1, batch_size=5, verbose=1)
    seed = 7
    np.random.seed(seed)
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(estimator, X, dummy_y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    #Predictions
    model = baseline_model()
    preds = model.predict(X)

    print(preds)



if False:
    dataset_path = '/Users/juanerolon/Desktop/'
    df = pd.read_csv(dataset_path + "sonar.csv", header=None)

    print(df.columns)
    print(df.head())

    dataset = df.values
    Y = dataset[:, 60]
    X = dataset[:, 0:60].astype(float)
    print(dataset)
    print("")
    print(Y)
    print("")
    print(X)
