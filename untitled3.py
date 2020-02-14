# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 18:22:51 2020

@author: spriyadarshini
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:50:29 2020

@author: spriyadarshini
"""



import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np


data = pd.read_csv('train_fwYjLYX.csv')

def add_datepart(df, fldname, drop=True):
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld,
                                     infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek',
            'Dayofyear', 'Is_month_end', 'Is_month_start',
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end',
            'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt,n.lower())
#add_datepart(data,'application_date')

#data['application_date'] = pd.to_datetime(data['application_date'])

#data['year']  = pd.DatetimeIndex(data['application_date']).year
#data['month']  = pd.DatetimeIndex(data['application_date']).month

data['segment'].value_counts()

data.isna().sum()

total_cases_daily = pd.DataFrame(data = data.groupby(['application_date','segment'])['case_count'].sum())

df1 = pd.DataFrame(total_cases_daily.index.tolist(), columns = ['application_date','segment'])

new_df = pd.concat([s.reset_index(drop=True) for s in [total_cases_daily, df1]], axis=1)

def dataframe_ready(df):
    add_datepart(df,'application_date')
    df.drop(columns=['application_date'],axis =1,inplace = True)
        
dataframe_ready(new_df)
y = new_df['case_count']
X = new_df.drop(columns = ['case_count'],axis = 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 42)

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

"""from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300,random_state = 42)"""  #599
rf_random.fit(X_train,y_train)
rf_random.best_params_

y_pred = rf_random.predict(X_test)

def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100
mean_absolute_percentage_error(y_test,y_pred)  #582.2795765393026
