# q1
#%%
import pandas as pd 
import numpy as np 
import math

#%%
e1 = math.e**(-9/2)
c = 9/2
kc1 = e1*(c)**5/math.factorial(5)/3**5

kc2 = e1*(c)**4/math.factorial(4)*math.comb(4,1)/(3**4)
kc3 = e1*(c)**3/math.factorial(3)*math.comb(3,1)/(3**3)
kc4 = e1*(c)**1/math.factorial(1)/math.comb(3,1)


kc1+kc2+kc3+kc4



# %%
