# Random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Ads_CTR_Optimisation.csv')

import random

N=10000
d=10
total_reward=0
ads_selected=[]

for i in range(0,N):
    ad=random.randrange(d)
    ads_selected.append(ad)
    reward=dataset.values[i,ad]
    total_reward=total_reward+reward
    
    
#Visualing results
plt.hist(ads_selected,color='blue')
plt.show()


    
