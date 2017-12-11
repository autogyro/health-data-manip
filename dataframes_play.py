#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:02:43 2017
@author: juanerolon
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Random sample of reals in [a,b] (b - a) * random_sample() + a
#
#Three-by-two array of random numbers from [-5, 0)
# 5 * np.random.random_sample((3, 2)) - 5
#

a, b = 1.0, 10.0

data = a + (b-a)*np.random.random_sample(10)
data2 = np.random.randint(2, size=10)

d = {'sample': data, 'value': data2}

df = pd.DataFrame(d)

max = np.round(np.max(df['sample'].values),4)
min = np.round(np.min(df['sample'].values),4)
rango = np.round((max-min),4)
bin_size = 2.0
print("Dataframe:\n")
print(df.head(10))
print("")
print("max: {}, min: {}, range: {}\n".format(max, min, rango))
print("Ceiling max: {}".format(np.ceil(max)))
print("Floored min: {}".format(np.floor(min)))

bins = []
span = np.floor(min)

while span <= np.ceil(max) + bin_size:
    bins.append(span)
    span += bin_size

bins = np.round(bins, 2)
print(bins)
print("")
#----------------------------------------------

bins_str = []
for m in range(len(bins) -1 ):
    label = "{} to {}".format(bins[m], bins[m+1])
    bins_str.append(label)
print(bins_str)
print("")
print("-----------------------------------------")
#-----------------------------------------------

# feat = 'sample'
# for m in range(len(bins)-1):
#     df1 = df[(df[feat] >= bins[m])]
#     df2 = df1[df1[feat]< bins[m+1]]
#     nv =  df2[feat].count()
#     print(bins_str[m] + ": {}".format(nv))
#-----------------------------------------------


feat = 'sample'
feat2 = 'value'
s1_array = []
s0_array = []
for m in range(len(bins)-1):
    df1 = df[(df[feat] >= bins[m])]
    df2 = df1[df1[feat] < bins[m+1]]
    nv =  df2[feat].count()

    print(df2)
    print("Bin: " + bins_str[m])
    print("Sample count: {}".format(nv))
    print("")
    s1 = df2[df2[feat2] == 1][feat2].count()
    s0 = df2[df2[feat2] == 0][feat2].count()
    s1_array.append(s1)
    s0_array.append(s0)

    print("s1 = {}, s0={}".format(s1,s0))
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