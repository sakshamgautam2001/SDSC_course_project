#Random forest regression

#import libs
import pandas as pa
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset=pa.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#no feature scaling is required

#creating random forest regressor
from sklearn.ensemble import RandomForestRegressor
Regressor=RandomForestRegressor(n_estimators=500,random_state=0)
Regressor.fit(X,y)

y_pred=Regressor.predict([[6.5]])

#making plot
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)

plt.scatter(X,y,color='red')
plt.plot(X_grid,Regressor.predict(X_grid),color='blue')
plt.show
