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

#####################
###### Analysis ######
#####################

###############
## regression##
###############

#again

X = df3.drop('MEDV', axis = 1)
y = df3['MEDV']

import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

reg = linear_model.LinearRegression()

reg.fit(X, y)

reg.score(X,y)
# 0.92% for the r^2, not bad!


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

reg = LinearRegression()

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

reg.score(X_test, y_test)

print("R^2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

#################################
## retrieving the ANCOVA output ##
#################################

# like in most social sciences

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


X = df3.drop('MEDV', axis = 1)
y = df3['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

## with all variables

model1 = sm.OLS(y, X)

results = model1.fit()

# for the output of the anova table :D
print(results.summary())
## adjusted R^2 of 0.99!

##########################
# reguralized regression #
##########################

from sklearn.linear_model import Ridge

X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size = 0.3, random_state=42)

ridge = Ridge(alpha = 0.1, normalize  = True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)


reg.score(X_test, y_test)
# still 90.17%

print("R^2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

######################
## Lasso regression ##
######################

from sklearn.linear_model import Lasso

lasso = Lasso (alpha = 0.1)

lasso.fit(X1_train, y1_train)
ridge_pred = lasso.predict(X1_test)


lasso.score(X1_test, y1_test)
# still 90.29%

print("R^2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

## plotting the impact of the variables

names = list(X.columns) 
print(names)

lasso_coef = lasso.fit(X, y).coef_

names = list(X1.columns) 
print(names)

lasso_coef = lasso.fit(X1, Y1).coef

plt.plot(names, lasso_coef)
#plt.xlabel(names)
plt.xticks(range(len(names)), names, rotation = 90)
plt.show()

################################################################
# regrssing with only the most important predictive variables #
################################################################

better_list = ["CRIM", "CHAS", "AGE", "B", "LSTAT", "TAX", "MEDV"]

df4 = df3[better_list]

df4.info()

######################
## regressing again ##
######################

reg2 = LinearRegression()


X2 = df4.drop('MEDV', axis = 1)
y2 = df4['MEDV']


X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.3, random_state=42)

reg2.fit(X2_train, y2_train)
reg2.score(X2_train, y2_train)

# score is getting substantially worse! Those variables should not be excluded!

