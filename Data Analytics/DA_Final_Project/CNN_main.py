
# ID: D10546004
# 2022 DA Final Project - Cervical cancer analysis

from zipfile import ZipFile
import numpy as np
import pandas as pd
from matplotlib import image
import matplotlib.pyplot as plt
# from scipy import stats
import os
import re
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from collections import Counter
# import myCNN
# from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!
import torchvision.io as tio
from torchvision.io import read_image
import PIL
# import torchvision.transforms.functional as transform
# import cv2
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
import torch
from torch import nn  # All neural network modules
from torch import optim  # For optimizers like SGD, Adam, etc.
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from torch.utils.data import Dataset


# the max pixels is 114X116 pixels. It needs to resize with interpolation mathod.


# get current working directory
cwd= os.getcwd()

# add the read directory to the path
rd = os.path.join(cwd)
zip_file = ZipFile('cropped.zip')

# Reads a file using pillow
dfs = {png_file.filename: 
        TF.to_tensor(
        # np.array(
            TF.resize(PIL.Image.open(zip_file.open(png_file.filename))
            ,size=[28,28] # CNN need to have idendical size. Size matters?
            )
            )
       for png_file in zip_file.infolist()
       if png_file.filename.endswith('.png')}
#
png_df = pd.DataFrame([dfs])



# define gender by hand, each value for 10 pictures
col_name = list(png_df.columns)
cell_list = []
shape_list = []
for i in col_name:
    j= re.split('[\/, \d*]', i)[1]
    cell_list.append(j)
    shape_list.append(png_df.loc[0,i].shape)

# np.array([i[1] for i in shape_list]).max() # 141 to 28
# np.array([i[2] for i in shape_list]).max() # 116 to 28 

cell_df = png_df.copy()
cell_df.columns = cell_list
cell_df.columns


unique_cell = [k for k,v in Counter(cell_list).items()]



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):

  def __init__(self,df):

    self.x=df[0].values
    self.x_train = torch.from_numpy(np.stack(self.x))
    # self.x_train = torch.stack(self.x)
    unique_cell = [k for k,v in Counter(df.index).items()]
    y = np.array([unique_cell.index(i) for i in df.index])

    # self.x_train=torch.tensor(x,dtype=torch.float32)
    self.y_train=torch.tensor(y)

  def __len__(self):
    return len(self.y_train)
  
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]


# Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)


    model.train()
    return num_correct/num_samples



# Hyperparameters
in_channels = 3
num_classes = 5
learning_rate = 0.001
batch_size = 64
num_epochs = 10

train, test = train_test_split(cell_df.transpose(), test_size=0.2)
myDS = MyDataset(train)
train_loader=DataLoader(MyDataset(train),batch_size=batch_size,shuffle=True)
test_loader=DataLoader(MyDataset(test),batch_size=batch_size,shuffle=True)

model = CNN()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")    