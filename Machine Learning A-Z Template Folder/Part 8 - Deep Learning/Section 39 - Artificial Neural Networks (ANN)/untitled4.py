# import the libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,-1].values

# since it has categorical data. need encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_1 = LabelEncoder()
X[:,1] = le_1.fit_transform(X[:,1])
le_2 =  LabelEncoder()
X[:,2] = le_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# to avoid dummy trap
X = X[:,1:]

# scaling the data
from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()
X = standardscaler.fit_transform(X)

# separating training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0,test_size = 0.20)

# Now applying ANN model to the dataset
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#creating input layer and first hidden layer
classifier.add(Dense(input_dim = 11, activation = 'relu',init = 'uniform',output_dim = 6))

#creating second hidden layer
classifier.add(Dense(activation = 'relu',init = 'uniform',output_dim = 6))

#creating the output layer
classifier.add(Dense(activation = 'sigmoid',init = 'uniform',output_dim = 1))

#Compiling the classifier ie applying Schotistic gradient descent . adam is a algorithm based on Schotistic gradient descent
#loss function for binary output. if more than 2 it would have been categorical_crossentropy
classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'])

# now fitting the model the the train set
classifier.fit(X_train,y_train,batch_size = 10,epochs = 100)

#predicting
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_pred,y_test)
accuracy_score(y_pred,y_test)
