##### Multiple Linear Regression

### Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

### Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
# Encode independent variables
labelEncoder_X = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)

### Avoiding the Dummy Variable Trap
X = X[:,1:] # Not necessary for this library but it is for some

### Splitting into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

### Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

### Predicting the test set results
y_pred = regressor.predict(X_test)

### Building the optimal model using Backward Elimination
import statsmodels.api as sm
# Append column of 1s to beginning of X to account for c0
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# (Step 2) Fit the model with all possible predictors
regressor_OLS = sm.OLS(endog = y, exog = X_opt.astype('float64')).fit()
regressor_OLS.summary()

# Remove index 2
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt.astype('float64')).fit()
regressor_OLS.summary()
# Remove index 1
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt.astype('float64')).fit()
regressor_OLS.summary()
# Remove index 4
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt.astype('float64')).fit()
regressor_OLS.summary()
# Remove index 5
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt.astype('float64')).fit()
regressor_OLS.summary()