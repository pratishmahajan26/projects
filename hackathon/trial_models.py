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

#new_df.set_index('date')
#data.loc[data['application_date'] == '2019-07-24']
"""
segment1 = new_df.loc[new_df['segment'] == 1]
segment1.drop('segment', inplace = True, axis = 1)
segment1.set_index('application_date', inplace = False)

segment2 = new_df.loc[new_df['segment'] == 2]
segment2.drop('segment', inplace = True, axis = 1)
segment2.set_index('application_date', inplace = False)

segment1.plot(figsize=(15, 6))
plt.show()

new_train_sample = pd.concat([segment1,segment2])
"""
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

######### using randomsearch  #######################
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



############################################################################
# now getting error for test data
test_data = pd.read_csv('test_1eLl9Yf.csv')
test_data.drop(columns = ['id'],axis = 1,inplace = True)
dataframe_ready(test_data)
test_data= scaler.transform(test_data)

y_pred2 = rf_random.predict(test_data)

sample_data = pd.read_csv('test_1eLl9Yf.csv')
sample_data['case_count'] = y_pred2

sample_data.to_csv('test_solution2.csv', index = False)



########################### LSTM  ################################################

import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np


data = pd.read_csv('train_fwYjLYX.csv')

data['segment'].value_counts()

data.isna().sum()

total_cases_daily = pd.DataFrame(data = data.groupby(['application_date','segment'])['case_count'].sum())

df1 = pd.DataFrame(total_cases_daily.index.tolist(), columns = ['application_date','segment'])

new_df = pd.concat([s.reset_index(drop=True) for s in [total_cases_daily, df1]], axis=1)

segment1 = new_df.loc[new_df['segment'] == 1]
segment1.drop('segment', inplace = True, axis = 1)
segment1.set_index('application_date', inplace = True)

segment2 = new_df.loc[new_df['segment'] == 2]
segment2.drop('segment', inplace = True, axis = 1)
segment2.set_index('application_date', inplace = True)

training_segment1 = segment1.iloc[:, 0:1].values
training_segment2 = segment2.iloc[:, 0:1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_segment1_scaled = sc.fit_transform(training_segment1)
training_segment2_scaled = sc.fit_transform(training_segment2)

# Creating a data structure with 60 timesteps and 1 output
segment1_X_train = []
segment1_y_train = []
for i in range(20, 806):
    segment1_X_train.append(training_segment1_scaled[i-20:i, 0])
    segment1_y_train.append(training_segment1_scaled[i, 0])
segment1_X_train, segment1_y_train = np.array(segment1_X_train), np.array(segment1_y_train)

segment2_X_train = []
segment2_y_train = []
#len(training_segment2_scaled)
for i in range(20, 844):
    segment2_X_train.append(training_segment2_scaled[i-20:i, 0])
    segment2_y_train.append(training_segment2_scaled[i, 0])
segment2_X_train, segment2_y_train = np.array(segment2_X_train), np.array(segment2_y_train)

# Reshaping
segment1_X_train = np.reshape(segment1_X_train, (segment1_X_train.shape[0], segment1_X_train.shape[1], 1))
segment2_X_train = np.reshape(segment2_X_train, (segment2_X_train.shape[0], segment2_X_train.shape[1], 1))

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (segment1_X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_absolute_error')
# Fitting the RNN to the Training set
regressor.fit(segment1_X_train, segment1_y_train, epochs = 100, batch_size = 32)
############################segment 2####################################

regressor1 = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor1.add(LSTM(units = 50, return_sequences = True, input_shape = (segment2_X_train.shape[1], 1)))
regressor1.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor1.add(LSTM(units = 50, return_sequences = True))
regressor1.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor1.add(LSTM(units = 50, return_sequences = True))
regressor1.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor1.add(LSTM(units = 50))
regressor1.add(Dropout(0.2))

# Adding the output layer
regressor1.add(Dense(units = 1))

# Compiling the RNN
regressor1.compile(optimizer = 'adam', loss = 'mean_absolute_error')
# Fitting the RNN to the Training set
regressor1.fit(segment1_X_train, segment1_y_train, epochs = 100, batch_size = 32)



###################################################################################
dataset_test = pd.read_csv('test_1eLl9Yf.csv')
segment1_test = dataset_test.loc[dataset_test['segment'] == 1]  #87
#806+87
#dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
#893-87-20
inputs = training_segment1_scaled[786:]
inputs = inputs.reshape(-1,1)
#inputs = sc.transform(inputs)
segment1_test = dataset_test.loc[dataset_test['segment'] == 1]
X_test = []
for i in range(20, 107):
    X_test.append(inputs[i-20:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = sc.inverse_transform(predicted_stock_price)



######################################segment2######################################
segment2_test = dataset_test.loc[dataset_test['segment'] == 2]   #93
inputs1 = training_segment2_scaled[824:]

inputs1 = inputs1.reshape(-1,1)
#inputs = sc.transform(inputs)
X_test1 = []
for i in range(20, 113):
    X_test1.append(training_segment1_scaled[i-20:i, 0])
X_test1 = np.array(X_test1)
X_test1 = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[1], 1))
predicted_stock_price = regressor.predict(X_test1)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

