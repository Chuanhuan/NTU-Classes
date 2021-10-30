
#%%
import pandas as pd 
import numpy as np 
import math
#%%

# q1-a

solution = 0
for n in range(1,7):
    tmp = math.comb(8,n-1)/math.comb(10,n)
    print('the i is',tmp)
    solution=tmp/6+solution
print(solution)

# q1-b expectation of x

solution = 0
for x in range(1,11):
    for n in range(1,7):
        tmp = x*math.comb(10-x,n-1)/math.comb(10,n)
        print('the i is',tmp)
        solution=tmp/6+solution
print(solution)



# %%
