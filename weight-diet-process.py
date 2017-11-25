#Testing loading datasets
#@Juan E. Rolon
#https://github.com/juanerolon

import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#path to datasets folders
datasets_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/datasets/'
project_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/capstone/gits/health-data-manip/'

#Import weight and nutrition dataset
weight_data = pd.read_sas(datasets_path + 'weight_history/WHQ_H.XPT')
nutrition_data = pd.read_sas(datasets_path + 'diet_nutrition/DBQ_H.XPT')

#Import alcohol-age cleaned dataset using Apache Arrow feather format
import feather
alcohol_smoking_data = feather.read_dataframe(project_path + 'alcohol_smoking_data.feather')
alcohol_smoking_data = alcohol_smoking_data.set_index('SEQNDX') #Restore the index saved during the export step