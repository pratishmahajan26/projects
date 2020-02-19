# import libs
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

# fitting reggressor to the model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,Y)

#Predict the salary of level 6.5
y_pred = regressor.predict(6.5)


#visualise the predication with higher resolution 
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color = 'r')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title('Position vs Salary (DecisionTreeRegressor)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()


# but not so useful with one independent variable. 