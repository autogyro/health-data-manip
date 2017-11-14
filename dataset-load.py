#Testing loading ASCII/TSV Files
#@Juan E. Rolon


import pandas as pd

data_dir_2 = '/Users/juanerolon/Downloads/ICPSR_25505_Demographics/DS0001/'
data_dir_3 = '/Users/juanerolon/Downloads/ICPSR_25505_Diabetes/DS0209/'
data_dir_4 = '/Users/juanerolon/Downloads/ICPSR_25505_Alcohol_Use/DS0202/'


data2 = pd.read_csv(data_dir_2 + '25505-0001-Data.tsv', sep='\t')
data3 = pd.read_csv(data_dir_3 + '25505-0209-Data.tsv', sep='\t')
data4 = pd.read_csv(data_dir_4 + '25505-0202-Data.tsv', sep='\t')

demographics_keys = list(data2.head().keys())
diabetes_keys = list(data3.head().keys())
alcohol_keys = list(data4.head().keys())

print('Different keys: diabetes-demographics\n')
print(list(set(diabetes_keys)-set(demographics_keys)))
print('Common keys: diabetes-demographics\n')
print(list(set(diabetes_keys) & set(demographics_keys)))
print("\n\n")
print('Different keys: alcohol-demographics\n')
print(list(set(alcohol_keys)-set(demographics_keys)))
print('Common keys: alcohol-demographics\n')
print(list(set(alcohol_keys) & set(demographics_keys)))

nkeys_demographics = len(demographics_keys)
nkeys_diabetes = len(diabetes_keys)
nkeys_diabetes = len(alcohol_keys )


# RIAGENDR_VALS2 = data2['RIAGENDR'].head()
# print(RIAGENDR_VALS2)
# print("---")
# RIAGENDR_VALS3 = data3['RIAGENDR'].head()
# print(RIAGENDR_VALS3)
