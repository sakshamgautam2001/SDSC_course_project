
#Import libs

import pandas as pa
import matplotlib.pyplot as plt
import numpy as np

#Import dataset
dataset=pa.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Linear Regression
from sklearn.linear_model import LinearRegression
Reg=LinearRegression()
Reg.fit(X,y)

#Fitting polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X)

Reg_2=LinearRegression().fit(X_poly,y)

#Linear plot
plt.scatter(X,y,color='red')
plt.plot(X,Reg.predict(X),color='blue')
plt.show()

#Polynomial plot
plt.scatter(X,y,color='red')
plt.plot(X,Reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,Reg_2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()






