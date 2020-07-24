#importing libs

import pandas as pa
import matplotlib.pyplot as plt
import numpy as np

#import dataset
dataset=pa.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values



#feature scaling
'''     from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y.reshape(-1,1))    '''


#Regressor
from sklearn.tree import DecisionTreeRegressor
Regressor=DecisionTreeRegressor(random_state=0)
Regressor.fit(X,y)

y_pred=Regressor.predict([[6.5]])

#Feature scaling should be done


#plot
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,Regressor.predict(X_grid),color='blue')
plt.show()


