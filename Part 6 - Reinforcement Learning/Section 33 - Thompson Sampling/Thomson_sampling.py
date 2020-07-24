# Upper confidence bound

#data preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Ads_CTR_Optimisation.csv')

import random

N=10000
d=10
total_reward=0
ads_selected=[]
no_of_1=[0]*d
no_of_0=[0]*d

for i in range(0,N):
    max_random=0
    ad=0
    for j in range(0,d):
        rand=random.betavariate(no_of_1[j]+1,no_of_0[j]+1)
            
        if(rand > max_random):
            max_random=rand
            ad=j
    
    ads_selected.append(ad)
    reward=dataset.values[i,ad]
    total_reward=total_reward+reward
    
    if(reward==0):
        no_of_0[ad]=no_of_0[ad]+1
    else:
        no_of_1[ad]=no_of_1[ad]+1
        

    
    

    
#Visualing results
plt.hist(ads_selected,color='blue')
plt.show()



