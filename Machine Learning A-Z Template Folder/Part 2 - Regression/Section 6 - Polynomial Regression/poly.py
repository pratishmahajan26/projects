#import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#import dataset
dataset  = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Fitting linear regression to dataset   
# trained using all the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)


#visualising linear regression
plt.scatter(X,y,color='Red')
plt.plot(X,lin_reg.predict(X),color = 'Blue')
plt.title('level vs salary (Linear regression)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()


#visualising polynomial regression
plt.scatter(X,y,color='Red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color = 'Blue')
plt.title('level vs salary (Polynomial regression)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

#visualising polynomial regression with more grids
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y,color='Red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color = 'Blue')
plt.title('level vs salary (Polynomial regression with grids)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()


#predicting salary of employee using linear regression
lin_reg.predict(6.5)

#predicting salary of employee using polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))