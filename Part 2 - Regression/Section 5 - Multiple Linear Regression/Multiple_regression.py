
#Import libs

import pandas as pa
import matplotlib.pyplot as plt
import numpy as np

#Import dataset
dataset=pa.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#Label encoding of data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#Avoiding dummy variable trap
X=X[:,1:]

#Dividing into test and train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8)

#Regression on test values
from sklearn.linear_model import LinearRegression
Regressor=LinearRegression()
Regressor.fit(X_train,y_train)

#Predicting values
y_pred=Regressor.predict(X_test)

#Backward Elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones([50,1]).astype(int),axis=1,values=X)
X_opt=X[:,[0,1,2,3,4,5]]
Regressor_OLS=sm.ols(endog=y, exog=X_opt).fit()


