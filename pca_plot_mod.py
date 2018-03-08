
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

print("Num. features in biochemistry data: {}".format(len(bio_cols)))
print("Num. features in questionnaire data: {}".format(len(que_cols)))


biochemistry_data = full_data[bio_cols]
questionnaire_data = full_data[que_cols]



good_data = biochemistry_data[['LBXSATSI', 'LBXSBU','LBXSCK','LBXSGTSI','LBXSIR','LBXSTR']]


#good_data = biochemistry_data[['LBXSAPSI','LBXSASSI','LBXSATSI','LBXSBU','LBXSCK','LBXSCR',
#                               'LBXSGTSI','LBXSIR', 'LBXSTB','LBXSTR','LBXSUA']]

#Standard PCA decomposition
if True:

    from sklearn.decomposition import PCA
    pca = PCA(n_components=good_data.shape[1])
    pca.fit(good_data)

    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1, len(pca.components_) + 1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns=good_data.keys())
    components.index = dimensions


    # PCA explained variance
    bare_ratios = pca.explained_variance_ratio_
    ratios = bare_ratios.reshape(len(pca.components_), 1)

    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns=['Explained Variance'])
    variance_ratios.index = dimensions

    if False:
        print("\nBare Explained Variance Ratios\n {}".format(pca.explained_variance_ratio_))
        print("\nExplained Variance Ratios DF\n {}".format(variance_ratios))
        print("\nDimensions\n {}".format(dimensions))
        print("\nComponents DF\n {}".format(components))

    nplots = 3
    sub_components = []
    sub_dimensions = []
    sub_var_ratios = []


    for i in range(nplots):

        fig, ax = plt.subplots(figsize=(14, 8))

        sc = components.loc[components.index[2*i:2*(i+1)]]
        vr = bare_ratios[2 * i:2 * (i + 1)]
        ix = sc.index

        sub_components.append(sc)
        sub_var_ratios.append(vr)
        sub_dimensions.append(ix)

        if False:
            print("\nSubcomponent DF[{}]:\n {}".format(i,sc))
            print("\nSub_var_ratios[{}]:\n {}".format(i,vr))
            print("\nDF Subdimensions[{}]:\n {}".format(i, ix))

        sc.plot(ax=ax, kind='bar');
        ax.set_ylabel("Feature Weights")
        ax.set_xticklabels(sc.index, rotation=0)

        for j, ev in enumerate(vr):
            ax.text(j - 0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n %.4f" % (ev))

        plt.show()
