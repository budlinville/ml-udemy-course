# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # make sure X is matrix
y = dataset.iloc[:, 2].values

# Splitting into training set and test set
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Feature scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # already fitted in previous line. Don't need to do again
"""

# Fitting the regression model to the dataset
# Create your regressor here

# Predicting a new result with Polynomial Regression
y_pred = regressor.predict(6.5)

# Visualizing the regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualizing the regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)     # gives vector
X_grid = X_grid.reshape((len(X_grid), 1))   # transform vector into matrix
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()