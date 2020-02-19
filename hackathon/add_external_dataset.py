# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 19:39:02 2020

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
        
data['segment'].value_counts()
data.isna().sum()
total_cases_daily = pd.DataFrame(data = data.groupby(['application_date','segment'])['case_count'].sum())
df1 = pd.DataFrame(total_cases_daily.index.tolist(), columns = ['application_date','segment'])
new_df = pd.concat([s.reset_index(drop=True) for s in [total_cases_daily, df1]], axis=1)

def add_gdp(df):
    gdp = []
    for x in df['application_date']:
        if x >= '2017-04-01' or x <= '2017-06-30':
            gdp.append(5.9)
        elif x >='2017-07-01' or x <= '2017-09-30':
            gdp.append(6.8)
        elif x>= '2017-10-01' or x <= '2017-12-31':
            gdp.append(7.7)
        elif x>= '2018-01-01' or x <= '2018-03-31':
            gdp.append(8.1)
        elif x >= '2018-04-01' or x <= '2018-06-30':
            gdp.append(8.0)
        elif x >='2018-07-01' or x <= '2018-09-30':
            gdp.append(7.0)
        elif x>= '2018-10-01' or x <= '2018-12-31':
            gdp.append(6.6)
        elif x>= '2019-01-01' or x <= '2019-03-31':
            gdp.append(5.8)
        elif x >= '2019-04-01' or x <= '2019-06-30':
            gdp.append(5.0)
        elif x >='2019-07-01' or x <= '2019-09-30':
            gdp.append(4.5)
        elif x>= '2019-10-01' or x <= '2019-12-31':
            gdp.append(6.2)
    df['gdp'] = gdp 
    
holiday_week =['2017-03-11','2017-03-12','2017-03-13','2017-06-20','2017-06-21','2017-06-22','2017-06-23','2017-06-24','2017-06-25','2017-09-21','2017-09-22','2017-09-23','2017-09-24','2017-09-25','2017-09-26','2017-09-27','2017-09-28','2017-09-29','2017-09-30','2017-10-15','2017-10-16','2017-10-17','2017-10-18',
               '2017-10-19','2017-12-20','2017-12-21','2017-12-22','2017-12-23','2017-12-24','2017-12-25','2018-02-26','2018-02-28','2018-02-27','2018-03-01','2018-03-02','2018-06-11','2018-06-12','2018-06-13','2018-06-14','2018-06-15','2018-06-16','2018-09-10','2018-09-11','2018-09-12','2018-09-13','2018-09-14','2018-09-15','2018-09-16','2018-09-17','2018-09-18','2018-09-19','2018-11-02','2018-11-03','2018-11-04','2018-11-05','2018-11-06','2018-11-07','2018-12-20','2018-12-21','2018-12-22','2018-12-23','2018-12-24','2018-12-25','2019-03-16','2019-03-17','2019-03-18','2019-03-19','2019-03-20','2019-03-21','2019-06-01','2019-06-02','2019-06-03','2019-06-04','2019-06-05','2019-08-30','2019-08-31','2019-09-01','2019-09-02','2019-09-03','2019-09-04','2019-09-05','2019-09-06','2019-09-07','2019-09-08','2019-10-25','2019-10-26','2019-10-27','2019-12-20','2019-12-21','2019-12-22','2019-12-23','2019-12-24','2019-12-25']
new_df['is_holiday_week']= [1 if x in holiday_week else 0 for x in new_df['application_date'] ]
add_gdp(new_df)
def dataframe_ready(df):
    add_datepart(df,'application_date')
    df.drop(columns=['application_date'],axis =1,inplace = True)
    
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


test_data = pd.read_csv('test_1eLl9Yf.csv')
test_data.drop(columns = ['id'],axis = 1,inplace = True)
sample = pd.read_csv('test_1eLl9Yf.csv')

######################################## segment1 ###################################################
segment1 = new_df.loc[new_df['segment'] == 1]
segment1.drop('segment', inplace = True, axis = 1)
dataframe_ready(segment1)
y1 = segment1['case_count']
X1 = segment1.drop(columns = ['case_count'],axis = 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X1=scaler.fit_transform(X1)

from sklearn.model_selection import train_test_split
X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y1,test_size = 0.20,random_state = 42)

from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
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
rf_random.fit(X1_train,y1_train)
rf_random.best_params_

y1_pred = rf_random.predict(X1_test)
mean_absolute_percentage_error(y1_test,y1_pred)  #582.2795765393026

test_data1 = test_data.loc[test_data['segment'] == 1]
test_data1.drop('segment', inplace = True, axis = 1)
add_gdp(test_data1) 
test_data1['is_holiday_week']= [1 if x in holiday_week else 0 for x in test_data1['application_date'] ]
dataframe_ready(test_data1)

test_data1= scaler.transform(test_data1)
y1_pred_segment1 = rf_random.predict(test_data1)

###################################### segment2 #######################################

segment2 = new_df.loc[new_df['segment'] == 2]
segment2.drop('segment', inplace = True, axis = 1)
dataframe_ready(segment2)
y2 = segment2['case_count']
X2 = segment2.drop(columns = ['case_count'],axis = 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X2=scaler.fit_transform(X2)

from sklearn.model_selection import train_test_split
X2_train,X2_test,y2_train,y2_test = train_test_split(X2,y2,test_size = 0.20,random_state = 42)

from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
from sklearn.ensemble import RandomForestRegressor
rf2 = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf2_random = RandomizedSearchCV(estimator = rf2, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
"""from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300,random_state = 42)"""  #599
rf2_random.fit(X2_train,y2_train)
rf2_random.best_params_

y2_pred = rf2_random.predict(X2_test)
mean_absolute_percentage_error(y2_test,y2_pred)  #582.2795765393026

test_data2 = test_data.loc[test_data['segment'] == 2]
test_data2.drop('segment', inplace = True, axis = 1)
add_gdp(test_data2) 
test_data2['is_holiday_week']= [1 if x in holiday_week else 0 for x in test_data2['application_date'] ]
dataframe_ready(test_data2)

test_data2= scaler.transform(test_data2)
y2_pred_segment2 = rf2_random.predict(test_data2)


################################## combing both ###################################

sample_data1 = sample.loc[sample['segment'] == 1] 
sample_data1 = sample_data1.set_index('application_date')
sample_data1['case_count'] = y1_pred_segment1

sample_data2 = sample.loc[sample['segment'] == 2] 
sample_data2 = sample_data2.set_index('application_date')
sample_data2['case_count'] = y2_pred_segment2

new_test_sample = pd.concat([sample_data1,sample_data2])

new_test_sample.to_csv('test_solution_both_rf.csv')

