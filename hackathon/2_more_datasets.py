# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:08:09 2020

@author: spriyadarshini
"""
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import datetime


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

new_df['application_date']=[datetime.datetime.strptime(x, '%Y-%m-%d') for x in new_df['application_date'] ]
#https://www.ceicdata.com/en/indicator/india/real-gdp-growthdata.loc[data['application_date'] < '2017-06-30'] 
def add_gdp(df):
    gdp = []
    for x in df['application_date']:
        if x >= datetime.datetime.strptime('2017-04-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2017-06-30', '%Y-%m-%d'):
            gdp.append(5.9)
        elif x >=datetime.datetime.strptime('2017-07-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2017-09-30', '%Y-%m-%d'):
            gdp.append(6.8)
        elif x>= datetime.datetime.strptime('2017-10-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2017-12-31', '%Y-%m-%d'):
            gdp.append(7.7)
        elif x>= datetime.datetime.strptime('2018-01-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2018-03-31', '%Y-%m-%d'):
            gdp.append(8.1)
        elif x >= datetime.datetime.strptime('2018-04-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2018-06-30', '%Y-%m-%d'):
            gdp.append(8.0)
        elif x >=datetime.datetime.strptime('2018-07-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2018-09-30', '%Y-%m-%d'):
            gdp.append(7.0)
        elif x>= datetime.datetime.strptime('2018-10-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2018-12-31', '%Y-%m-%d'):
            gdp.append(6.6)
        elif x>= datetime.datetime.strptime('2019-01-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2019-03-31', '%Y-%m-%d'):
            gdp.append(5.8)
        elif x >= datetime.datetime.strptime('2019-04-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2019-06-30', '%Y-%m-%d'):
            gdp.append(5.0)
        elif x >=datetime.datetime.strptime('2019-07-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2019-09-30', '%Y-%m-%d'):
            gdp.append(4.5)
        elif x>= datetime.datetime.strptime('2019-10-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2019-12-31', '%Y-%m-%d'):
            gdp.append(6.2)
    df['gdp'] = gdp 
    
#https://unemploymentinindia.cmie.com/kommon/bin/sr.php?kall=wsttimeseries&index_code=050050000000&dtype=total
def add_unemployentrate(df):
    urate=[]
    for x in df['application_date']:
        if x >= datetime.datetime.strptime('2017-04-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2017-04-30', '%Y-%m-%d'):
            urate.append(3.9)
        if x >= datetime.datetime.strptime('2017-05-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2017-05-31', '%Y-%m-%d'):
            urate.append(4.0)
        if x >= datetime.datetime.strptime('2017-06-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2017-06-30', '%Y-%m-%d'):
            urate.append(4.1)
        if x >= datetime.datetime.strptime('2017-07-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2017-07-31', '%Y-%m-%d'):
            urate.append(3.4)
        if x >= datetime.datetime.strptime('2017-08-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2017-08-31', '%Y-%m-%d'):
            urate.append(4.1)
        if x >= datetime.datetime.strptime('2017-09-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2017-09-30', '%Y-%m-%d'):
            urate.append(4.6)
        if x >= datetime.datetime.strptime('2017-10-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2017-10-31', '%Y-%m-%d'):
            urate.append(5.0)
        if x >= datetime.datetime.strptime('2017-11-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2017-11-30', '%Y-%m-%d'):
            urate.append(4.7)
        if x >= datetime.datetime.strptime('2017-12-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2017-12-31', '%Y-%m-%d'):
            urate.append(4.7)
        if x >= datetime.datetime.strptime('2018-01-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2018-01-31', '%Y-%m-%d'):
            urate.append(5.0)
        if x >= datetime.datetime.strptime('2018-02-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2018-02-28', '%Y-%m-%d'):
            urate.append(5.9)
        if x >= datetime.datetime.strptime('2018-03-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2018-03-31', '%Y-%m-%d'):
            urate.append(6.0)
        if x >= datetime.datetime.strptime('2018-04-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2018-04-30', '%Y-%m-%d'):
            urate.append(5.6)
        if x >= datetime.datetime.strptime('2018-05-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2018-05-31', '%Y-%m-%d'):
            urate.append(5.1)
        if x >= datetime.datetime.strptime('2018-06-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2018-06-30', '%Y-%m-%d'):
            urate.append(5.8)
        if x >= datetime.datetime.strptime('2018-07-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2018-07-31', '%Y-%m-%d'):
            urate.append(5.7)
        if x >= datetime.datetime.strptime('2018-08-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2018-08-31', '%Y-%m-%d'):
            urate.append(6.3)
        if x >= datetime.datetime.strptime('2018-09-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2018-09-30', '%Y-%m-%d'):
            urate.append(6.5)
        if x >= datetime.datetime.strptime('2018-10-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2018-10-31', '%Y-%m-%d'):
            urate.append(6.8)
        if x >= datetime.datetime.strptime('2018-11-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2018-11-30', '%Y-%m-%d'):
            urate.append(6.7)
        if x >= datetime.datetime.strptime('2018-12-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2018-12-31', '%Y-%m-%d'):
            urate.append(7.0)
        if x >= datetime.datetime.strptime('2019-01-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2019-01-31', '%Y-%m-%d'):
            urate.append(6.9)
        if x >= datetime.datetime.strptime('2019-02-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2019-02-28', '%Y-%m-%d'):
            urate.append(7.2)
        if x >= datetime.datetime.strptime('2019-03-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2019-03-31', '%Y-%m-%d'):
            urate.append(6.7)
        if x >= datetime.datetime.strptime('2019-04-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2019-04-30', '%Y-%m-%d'):
            urate.append(7.3)
        if x >= datetime.datetime.strptime('2019-05-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2019-05-31', '%Y-%m-%d'):
            urate.append(7.0)
        if x >= datetime.datetime.strptime('2019-06-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2019-06-30', '%Y-%m-%d'):
            urate.append(7.9)
        if x >= datetime.datetime.strptime('2019-07-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2019-07-31', '%Y-%m-%d'):
            urate.append(7.3)
        if x >= datetime.datetime.strptime('2019-08-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2019-08-31', '%Y-%m-%d'):
            urate.append(8.2)
        if x >= datetime.datetime.strptime('2019-09-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2019-09-30', '%Y-%m-%d'):
            urate.append(7.2)
        if x >= datetime.datetime.strptime('2019-10-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2019-10-31', '%Y-%m-%d'):
            urate.append(8.1)
        if x >= datetime.datetime.strptime('2019-11-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2019-11-30', '%Y-%m-%d'):
            urate.append(7.2)
        if x >= datetime.datetime.strptime('2019-12-01', '%Y-%m-%d') and x <= datetime.datetime.strptime('2019-12-31', '%Y-%m-%d'):
            urate.append(7.6)
    df['unemploymentrate'] = urate
    
holiday_list =['2017-03-13','2017-06-25','2017-09-30','2017-10-19','2017-12-25','2018-03-02','2018-06-16','2018-09-19','2018-11-07','2018-12-25','2019-03-21','2019-06-05','2019-09-08','2019-09-27','2019-12-25']
holiday_week1 =['2017-03-11','2017-03-12','2017-03-13','2017-06-23','2017-06-24','2017-06-25','2017-09-21','2017-09-22','2017-09-23','2017-09-24','2017-09-25','2017-09-26','2017-09-27','2017-09-28','2017-09-29','2017-09-30','2017-10-17','2017-10-18','2017-10-19','2017-12-20','2017-12-21','2017-12-22','2017-12-23','2017-12-24','2017-12-25','2018-03-01','2018-03-02','2018-06-13','2018-06-14','2018-06-15','2018-06-16','2018-09-10','2018-09-11','2018-09-12','2018-09-13','2018-09-14','2018-09-15','2018-09-16','2018-09-17','2018-09-18','2018-09-19','2018-11-05','2018-11-06','2018-11-07','2018-12-20','2018-12-21','2018-12-22','2018-12-23','2018-12-24','2018-12-25','2019-03-19','2019-03-20','2019-03-21','2019-06-02','2019-06-03','2019-06-04','2019-06-05','2019-08-30','2019-08-31','2019-09-01','2019-09-02','2019-09-03','2019-09-04','2019-09-05','2019-09-06','2019-09-07','2019-09-08','2019-10-25','2019-10-26','2019-10-27','2019-12-20','2019-12-21','2019-12-22','2019-12-23','2019-12-24','2019-12-25']
holiday_week =['2017-03-11','2017-03-12','2017-03-13','2017-06-20','2017-06-21','2017-06-22','2017-06-23','2017-06-24','2017-06-25','2017-09-21','2017-09-22','2017-09-23','2017-09-24','2017-09-25','2017-09-26','2017-09-27','2017-09-28','2017-09-29','2017-09-30','2017-10-15','2017-10-16','2017-10-17','2017-10-18',
               '2017-10-19','2017-12-20','2017-12-21','2017-12-22','2017-12-23','2017-12-24','2017-12-25','2018-02-26','2018-02-28','2018-02-27','2018-03-01','2018-03-02','2018-06-11','2018-06-12','2018-06-13','2018-06-14','2018-06-15','2018-06-16','2018-09-10','2018-09-11','2018-09-12','2018-09-13','2018-09-14','2018-09-15','2018-09-16','2018-09-17','2018-09-18','2018-09-19','2018-11-02','2018-11-03','2018-11-04','2018-11-05','2018-11-06','2018-11-07','2018-12-20','2018-12-21','2018-12-22','2018-12-23','2018-12-24','2018-12-25','2019-03-16','2019-03-17','2019-03-18','2019-03-19','2019-03-20','2019-03-21','2019-06-01','2019-06-02','2019-06-03','2019-06-04','2019-06-05','2019-08-30','2019-08-31','2019-09-01','2019-09-02','2019-09-03','2019-09-04','2019-09-05','2019-09-06','2019-09-07','2019-09-08','2019-10-25','2019-10-26','2019-10-27','2019-12-20','2019-12-21','2019-12-22','2019-12-23','2019-12-24','2019-12-25']
add_gdp(new_df)
add_unemployentrate(new_df)
new_df['is_holiday_week']= [1 if x in holiday_week else 0 for x in new_df['application_date'] ]

def dataframe_ready(df):
    add_datepart(df,'application_date')
    df.drop(columns=['application_date'],axis =1,inplace = True)
        
dataframe_ready(new_df)
y = new_df['case_count']
X = new_df.drop(columns = ['case_count'],axis = 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X=scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 42)

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 300, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(90, 110, num = 11)]
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
               'bootstrap': bootstrap,
               }
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


test_data = pd.read_csv('test_1eLl9Yf.csv')
test_data.drop(columns = ['id'],axis = 1,inplace = True)
test_data['application_date']=[datetime.datetime.strptime(x, '%Y-%m-%d') for x in test_data['application_date'] ]

add_gdp(test_data)
add_unemployentrate (test_data)
test_data['is_holiday_week']= [1 if x in holiday_week else 0 for x in test_data['application_date'] ]
dataframe_ready(test_data)
test_data= scaler.transform(test_data)

y_pred2 = rf_random.predict(test_data)

sample_data = pd.read_csv('test_1eLl9Yf.csv')
sample_data['case_count'] = y_pred2
sample_data.to_csv('test_sol1.csv', index = False)

