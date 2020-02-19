# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 20:29:56 2019

@author: spriyadarshini
"""

# import libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,[1]].values
# scaling is a must . we will do normalisation 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = []
y_train = []
# now we will create dataset of timestamp 60 and 1 output ie one row will contain 60 days of stock price and 61st will be the predicted stock price
for i in range(60,training_set.shape[0]):
    seq = training_set[i-60:i]
    X_train.append(seq)
    y_train.append(training_set[i])
    
#since RNN only take array... we have to convert the lists into array
X_train = np.array(X_train)
y_train = np.array(y_train)

#building the model with 4 LSTM layers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout  # used for randomly dropping out the neurons to prevent overfitting

regressor = Sequential()
regressor.add(LSTM(units = 50,return_sequences = True, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(rate = 0.2))
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(rate = 0.2))
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(rate = 0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(rate = 0.2))

# output layer
regressor.add(Dense(output_dim = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train,y_train,batch_size = 32, epochs = 100)

#predicting and visualising the model
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,[1]].values

# now in order to predict the stock of 3rd jan 2017, we need to see the stocks of 60 days before that
#so we need to concatenate test and training dataset
dataset_total = pd.concat((dataset_train['Open'],dataset_train['Open']),axis = 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test) -60:].values
inputs = inputs.reshape(-1,1)
# since model was trained in scaled data.. 
inputs = sc.fit_transform(inputs)
X_test = []
for i in range(60,80):
    seq = inputs[i-60:i]
    X_test.append(seq)
X_test = np.array(X_test)

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# visualing the trend now
plt.plot(real_stock_price,color = 'red',label = 'Google real stock price')
plt.plot(predicted_stock_price,color = 'blue',label = 'Google predicted_stock_price')
plt.xlabel('time')
plt.ylabel('Google stock price')
plt.legend()
plt.show()


