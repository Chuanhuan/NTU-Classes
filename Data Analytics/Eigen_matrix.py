#%%
import numpy as np
import pandas as pd

m = np.matrix([[1,2,3],[4,5,6],[7,8,9,]])
print(m)

eig_value, eig_vector = np.linalg.eig(m)
print(eig_value,eig_vector)
# %%
eig_vector.dot(eig_vector.T)
# %%

# use to check the eigen vector is orthoganal is is S.P.D.
m = np.matrix([[1,2,3,4,1,2],[4,5,6,7,4,5],[7,8,9,9,3,4]])
cov_m = np.cov(m)
print(cov_m)
eig_value, eig_vector = np.linalg.eig(cov_m)
print('eigen value', eig_value)
print(eig_vector)
eig_vector.dot(eig_vector.T)

# %%
# find the first 2 eigen vector
eig_vector
eig_vector[:,:2]
# %%
new_m = np.zeros_like(eig_vector)
new_m[:,:2]=eig_vector[:,:2]
new_m
# %%
n = 4
P = np.random.randint(0,10,(n,n))

S = P @ P.T
print(S)

