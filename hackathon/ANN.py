# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:01:23 2020

@author: spriyadarshini
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:08:09 2020

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

#https://www.ceicdata.com/en/indicator/india/real-gdp-growthdata.loc[data['application_date'] < '2017-06-30'] 
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
holiday_list =['2017-03-13','2017-06-25','2017-09-30','2017-10-19','2017-12-25','2018-03-02','2018-06-16','2018-09-19','2018-11-07','2018-12-25','2019-03-21','2019-06-05','2019-09-08','2019-09-27','2019-12-25']
holiday_week1 =['2017-03-11','2017-03-12','2017-03-13','2017-06-23','2017-06-24','2017-06-25','2017-09-21','2017-09-22','2017-09-23','2017-09-24','2017-09-25','2017-09-26','2017-09-27','2017-09-28','2017-09-29','2017-09-30','2017-10-17','2017-10-18','2017-10-19','2017-12-20','2017-12-21','2017-12-22','2017-12-23','2017-12-24','2017-12-25','2018-03-01','2018-03-02','2018-06-13','2018-06-14','2018-06-15','2018-06-16','2018-09-10','2018-09-11','2018-09-12','2018-09-13','2018-09-14','2018-09-15','2018-09-16','2018-09-17','2018-09-18','2018-09-19','2018-11-05','2018-11-06','2018-11-07','2018-12-20','2018-12-21','2018-12-22','2018-12-23','2018-12-24','2018-12-25','2019-03-19','2019-03-20','2019-03-21','2019-06-02','2019-06-03','2019-06-04','2019-06-05','2019-08-30','2019-08-31','2019-09-01','2019-09-02','2019-09-03','2019-09-04','2019-09-05','2019-09-06','2019-09-07','2019-09-08','2019-10-25','2019-10-26','2019-10-27','2019-12-20','2019-12-21','2019-12-22','2019-12-23','2019-12-24','2019-12-25']
holiday_week =['2017-03-11','2017-03-12','2017-03-13','2017-06-20','2017-06-21','2017-06-22','2017-06-23','2017-06-24','2017-06-25','2017-09-21','2017-09-22','2017-09-23','2017-09-24','2017-09-25','2017-09-26','2017-09-27','2017-09-28','2017-09-29','2017-09-30','2017-10-15','2017-10-16','2017-10-17','2017-10-18',
               '2017-10-19','2017-12-20','2017-12-21','2017-12-22','2017-12-23','2017-12-24','2017-12-25','2018-02-26','2018-02-28','2018-02-27','2018-03-01','2018-03-02','2018-06-11','2018-06-12','2018-06-13','2018-06-14','2018-06-15','2018-06-16','2018-09-10','2018-09-11','2018-09-12','2018-09-13','2018-09-14','2018-09-15','2018-09-16','2018-09-17','2018-09-18','2018-09-19','2018-11-02','2018-11-03','2018-11-04','2018-11-05','2018-11-06','2018-11-07','2018-12-20','2018-12-21','2018-12-22','2018-12-23','2018-12-24','2018-12-25','2019-03-16','2019-03-17','2019-03-18','2019-03-19','2019-03-20','2019-03-21','2019-06-01','2019-06-02','2019-06-03','2019-06-04','2019-06-05','2019-08-30','2019-08-31','2019-09-01','2019-09-02','2019-09-03','2019-09-04','2019-09-05','2019-09-06','2019-09-07','2019-09-08','2019-10-25','2019-10-26','2019-10-27','2019-12-20','2019-12-21','2019-12-22','2019-12-23','2019-12-24','2019-12-25']
add_gdp(new_df)
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
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 42)

import keras
from keras.models import Sequential
from keras.layers import Dense

regressor = Sequential()
regressor.add(Dense(input_dim = 15, activation = 'relu',init = 'he_uniform',output_dim = 6))
regressor.add(Dense(activation = 'relu',init = 'he_uniform',output_dim = 6))
regressor.add(Dense(activation = 'sigmoid',init = 'glorot_uniform',output_dim = 1))
regressor.compile(optimizer = 'adam', loss= 'mae', metrics = ['accuracy'])
regressor.fit(X_train,y_train,batch_size = 8,epochs = 100)


y_pred = regressor.predict(X_test)

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
add_gdp(test_data) 
test_data['is_holiday_week']= [1 if x in holiday_week else 0 for x in test_data['application_date'] ]
dataframe_ready(test_data)
test_data= scaler.transform(test_data)

y_pred2 = regressor.predict(test_data)

sample_data = pd.read_csv('test_1eLl9Yf.csv')
sample_data['case_count'] = y_pred2

sample_data.to_csv('test_ann.csv', index = False)

