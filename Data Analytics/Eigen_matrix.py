#%%
import numpy as np
import pandas as pd

m = np.matrix([[1,2,3],[4,5,6],[7,8,9,]])
print(m)

eig_value, eig_vector = np.linalg.eig(m)
print(eig_value,eig_vector)
# %%
# two eig_vector product for what?
eig_vector.dot(eig_vector.T)
# %%

# use to check the eigen vector is orthoganalis S.P.D.
m = np.matrix([[1,2,3,4,1,2],[4,5,6,7,4,5],[7,8,9,9,3,4]])
cov_m = np.cov(m)
print('cov', cov_m)
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


# %%
evals, evecs = np.linalg.eig(S)
print(evals)

# %%

print('proof the two eigen vector orthoganal')
v1 = evecs[:,0] # First column is the first eigenvector
v2 = evecs[:,1] # Second column is the second eigenvector
v1.dot(v2) # same results
v1 @ v2
# %%
# the same result when S is symetric metrix
# not always true when S is not S.P.D.
S@ evecs  @evecs.T
S@ evecs  @np.linalg.inv(evecs)
# %%

# make the eigenvalues to a matrix
ss = int(evals.shape[0])
mm = np.zeros((ss,ss))

np.fill_diagonal(mm,evals)
print(mm)
# %%
m = np.matrix([[1,2,3,4,1,2],[4,5,6,7,4,5],[7,8,9,9,3,4]])
print(m.shape)
cov_m = np.cov(m)
print(cov_m)
print(np.cov(m.T,rowvar=0))
print(np.cov(m,rowvar=0))

# %%
# this not work becasue ndarray vs matrix is diff

print(m.shape)
m_bar = np.mean(m,axis=0)
print(m_bar)
print([float(i) for i in m_bar])
Center_mat = m - [float(i) for i in m_bar]

#%%
# use ndarray is more easier
import numpy as np

a1 = np.array([[1,2,3],[4,5,6]])
a1_bar = np.mean(a1,axis=0)
print(a1_bar)
print([float(i) for i in a1_bar])
Center_mat = a1 - [float(i) for i in a1_bar]
print(Center_mat)
print(a1 - a1_bar)
print(a1@a1.T)

#%%
# HW 6 FA
# Read Text Files with Pandas

col_names = ['mpg','cylinders','displacement','horsepower',
            'weight','acceleration','year','origin','car_name']
# read text file into pandas DataFrame
df = pd.read_fwf("auto-mpg.data.txt",header=None,names = col_names)
# df = pd.DataFrame(df1.to_numpy() , columns=col_names)
# display DataFrame
df = df[~df.isin({'?'}).any(1)]
print(df)


m = df.loc[:,['cylinders','displacement','horsepower',
            'weight','acceleration','year','origin']].astype(float).to_numpy()
result , eig_value , eig_vector = myPCA(m, isCorrMx = False)
print('Eigen value :',eig_value)
print('EigenVector size :',eig_vector.shape)


m_bar = np.mean(m.T,axis=1)
Center_mat = m - [float(i) for i in m_bar]

V = np.cov(Center_mat , rowvar =0)
eig_value, eig_vector = np.linalg.eig(V)
# print('Eigen value :',eig_value)
print('EigenVector size :',eig_vector.shape)


eig_size = int(eig_value.shape[0])
eigVal_m = np.zeros((eig_size,eig_size))
np.fill_diagonal(eigVal_m,eig_value)

print(eigVal_m.shape , eig_vector.shape)

q = 2
# update_eig_vector = np.zeros_like(eig_vector)
# update_eig_vector[:,:r] = eig_vector[:,:r]
update_eig_vector = eig_vector[:,:q]
print(update_eig_vector.shape)


# update_eig_value = np.zeros_like(eigVal_m)
# update_eig_value[:,:r] = eigVal_m[:,:r]
update_eig_value = eigVal_m[:q,:q]
print(update_eig_value.shape)

A_T = update_eig_vector@ np.sqrt(update_eig_value)
A =A_T.T
print('A: ', A.shape)

# h_sq = A_T@A
h_sq_m = update_eig_vector@update_eig_value@update_eig_vector.T
h_sq = h_sq_m.diagonal()
print('h_sq:',h_sq.shape)
# print(np.diagonal(h_sq))

psi = V - h_sq_m
print('psi:',  psi.shape)

tmp_term = np.linalg.inv(psi)@A_T
print('F:' ,m.shape, tmp_term.shape, A.shape)
F=m@tmp_term@np.linalg.inv(A@tmp_term)
print(F.shape)


l_var = []
total_var = V.trace()
for i in eig_value:
    l_var.append( np.round( i*100 /total_var , 3))
