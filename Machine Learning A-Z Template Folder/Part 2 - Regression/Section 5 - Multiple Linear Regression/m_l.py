#import the libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot

#importing the dataset
dataset = pd.read_csv('50_Startups.csv')

#separarting dependent and independent variable
#independent variables
X= dataset.iloc[:,:-1].values
#dependent variale
Y = dataset.iloc[:,4].values

#Encoding Categorical variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoid dummy variable trap ie to remove one variable
X = X[:,1:]

#separating training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

#Applying Linear regression to model ie training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# predicting the test result
Y_pred = regressor.predict(X_test)

#We will use Backward Elimination method to create the model
import statsmodels.formula.api as sm
#this module does not consider constant b0 in the linear equation. so we have to add a column x0 with value 1
X = np.append(arr = np.ones(shape = (50,1)), values = X ,axis = 1)

# creating the optimum X with First taking all independent variables
X_opt = X[:,[0,1,2,3,4,5]]

# fit the full model with all possible predictors
regressor_ols = sm.OLS(endog =Y , exog = X_opt).fit()
regressor_ols.summary()

# now after seeing the p_value of all variables , we need to remove,x2 with hightest p_value
X_opt = X[:,[0,1,3,4,5]]
regressor_ols = sm.OLS(endog =Y , exog = X_opt).fit()
regressor_ols.summary()

X_opt = X[:,[0,3,4,5]]
regressor_ols = sm.OLS(endog =Y , exog = X_opt).fit()
regressor_ols.summary()


X_opt = X[:,[0,3,5]]
regressor_ols = sm.OLS(endog =Y , exog = X_opt).fit()
regressor_ols.summary()

X_opt = X[:,[0,3]]
regressor_ols = sm.OLS(endog =Y , exog = X_opt).fit()
regressor_ols.summary()
