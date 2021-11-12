# q1
#%%
import pandas as pd 
import numpy as np 
import math

#%%

x = [[.2,.5,.1],[.4,.3,0],[.2,.1,.3]]
y = np.matrix(x).reshape(3,3)
print(y)
# %%
y**3
# %%
np.linalg.matrix_power(y,30)
# %%
# example 4.30

x = [[0,.4,0,0,0,0],
    [.6,0,.4,0,0,0],
    [0,.6,0,.4,0,0],
    [0,0,.6,0,.4,0],
    [0,0,0,.6,0,.4],
    [0,0,0,0,.6,0]]

y = np.matrix(x).reshape(6,6)
print(y)
np.linalg.matrix_power(y,10**3)
# %%
# limiting prob example 
x = [[.7,.3],[.4,.6]]
y = np.matrix(x).reshape(2,2)

output = np.linalg.matrix_power(y,100000).round(3).tolist()
np.linalg.matrix_power(y,1000)
# %%

# q2
x = [[.7,0,0,0,0,0,.3,0,0,0],
    [0,.2,0,0,0,.5,0,.3,0,0],
    [0,0,0,0,0,0,0,0,1,0],
    [.2,0,0,.2,.5,0,0,0,0,.1],
    [0,0,0,.4,.3,0,.1,0,.2,0],
    [0,.8,0,0,0,.1,0,.1,0,0],
    [.5,0,0,0,0,0,.5,0,0,0],
    [0,.4,0,0,0,.2,0,.4,0,0],
    [0,0,1,0,0,0,0,0,0,0],
    [0,.3,0,.2,.1,0,.1,0,0,.3]]

y = np.matrix(x).reshape(10,10)
# print(y)
output = np.linalg.matrix_power(y,10**6).round(3).tolist()

# %%
# try use all p to inverse
 np.linalg.inv(np.identity(10)-y)
# %%
