
# ID: D10546004
# 2022 DA Final Project - Cervical cancer analysis

from zipfile import ZipFile
import numpy as np
import pandas as pd
from matplotlib import image
import matplotlib.pyplot as plt
from scipy import stats
import os
import re
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from collections import Counter


# get current working directory
cwd= os.getcwd()

# add the read directory to the path
rd = os.path.join(cwd)
zip_file = ZipFile('cropped.zip')
dfs = {png_file.filename: image.imread(zip_file.open(png_file.filename))
       for png_file in zip_file.infolist()
       if png_file.filename.endswith('.png')}
png_df = pd.DataFrame([dfs])


# define gender by hand, each value for 10 pictures
col_name = list(png_df.columns)
cell_list = []
for i in col_name:
    j= re.split('[\/, \d*]', i)[1]
    cell_list.append(j)

unique_cell = [k for k,v in Counter(cell_list).items()]

# # explore transformation data
# png_df.iloc[0,1].shape # (84, 72, 3)
# tt = png_df.iloc[0,1].reshape(-1,3)
# tt.shape
# tt = tt.reshape(84,72,3)
# tt.shape
# plt.imshow(png_df.iloc[0,1])
# plt.imshow(tt)

fig, ax = plt.subplots(1, 5)
dic = {}          
k = 0
# insert all gender to new row
for i in unique_cell:    
    plt.subplot(1,5,k+1)
    plt.imshow(png_df.iloc[0,cell_list.index(i)])
    k = k +1

