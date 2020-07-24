# Upper confidence bound

#data preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Ads_CTR_Optimisation.csv')

import math

N=10000
d=10
total_reward=0
ads_selected=[]
no_of_selections=[0]*d
sum_of_rewards=[0]*d

for i in range(0,N):
    max_up_bound=0
    ad=0
    for j in range(0,d):
        if (no_of_selections[j]>0):
            avg_reward=sum_of_rewards[j]/no_of_selections[j]
            del_i=math.sqrt(3/2 * math.log(i+1)/no_of_selections[j])
            up_bound=avg_reward+del_i
            
        else:
            up_bound=1e400
            
        if(up_bound>max_up_bound):
            max_up_bound=up_bound
            ad=j
    
    ads_selected.append(ad)
    sum_of_rewards[ad]=sum_of_rewards[ad] + dataset.values[i,ad]
    no_of_selections[ad]=no_of_selections[ad]+1
    total_reward=total_reward + dataset.values[i,ad]
    
    

    
#Visualing results
plt.hist(ads_selected,color='blue')
plt.show()


