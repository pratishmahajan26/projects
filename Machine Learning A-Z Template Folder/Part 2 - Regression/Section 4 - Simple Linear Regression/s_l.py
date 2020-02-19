# Import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values      #independent variable
Y = dataset.iloc[:,1].values    #dependent variable

#splitting the training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3,random_state = 0)

#Fitting simple linear regression to the training set or training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#now when the model is trained we will what prediction it is making when passed test data
Y_pred = regressor.predict(X_test)

# now plotting the line using training set
plt.scatter(x=X_train,y=Y_train,color ='Red')
plt.plot(X_train,regressor.predict(X_train),color = 'Blue')
plt.title('Salary Vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# now plotting the line using test set
plt.scatter(x=X_test,y=Y_test,color = 'Red')
plt.plot(X_train,regressor.predict(X_train),color = 'Blue')   # the linear line should be same
plt.title('Salary Vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

