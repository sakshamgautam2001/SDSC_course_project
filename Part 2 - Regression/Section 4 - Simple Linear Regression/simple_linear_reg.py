#importing libs

import pandas as pa
import matplotlib.pyplot as plt
import numpy as np

#importing dataset

dataset=pa.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#Split into train and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3)

#Feature scaling is not required

#Regression

from sklearn.linear_model import LinearRegression
Regressor=LinearRegression()
Regressor.fit(X_train,y_train)

#Predicting test values
y_pred=Regressor.predict(X_test)

#Plotting the graphs
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,Regressor.predict(X_train),color='blue')
plt.title('training plot')
plt.show()

#Checking the test values
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,Regressor.predict(X_train),color='blue')
plt.show()







