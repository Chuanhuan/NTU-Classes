import numpy as np
import pandas as pd

m = np.matrix([[1,2,3,4],[4,5,6,7],[7,8,9,10]])
print(m)

eig_value, eig_vector = np.linalg.eig(m)
print(eig_value,eig_vector)