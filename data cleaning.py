# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 19:24:34 2019

@author: Nico
"""
#################################################################################
## Trying different machine learning algorithms on the boston housing data set ##
##################################################################################


import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df1 = pd.read_csv(r"C:\Users\Nico\Documents\Datasets\boston_1.csv", sep = "\,"   )

df1.head()

df1.info()


##########
## nans ##
##########

# Replacing them by the respective means

df1.isnull().values.any()
# there are some

lis1 = ["CRIM", "ZN", "INDUS", "CHAS", "AGE", "LSTAT"] # there are nans for these variables

for a in lis1:
    mean_value = df1[a].mean()
    df1[a] = df1[a].fillna(mean_value)


assert pd.notnull(df1).all().all()

df1.isnull().values.any()
## now: "false" --> all values were replaced!


###########################
## plotting the Variables #
###########################


 for a in lis1:
     plt.hist(df1[a])
     plt.title(a)
     plt.show()
     
import seaborn as sns

for a in lis1:    
    sns.boxplot(df1[a])


#######################
## dropping outliers ##
#######################
    
df2 = df1

df3 = df1[df1["ZN"] < 89]
df3 = df3[df3["INDUS"] < 24]
df3 = df3[df3["CHAS"] < 0.8]

####################
## all to float64 ##
####################

df3['RAD'] = df3['RAD'].astype('float64')
df3['TAX'] = df3['TAX'].astype('float64')


