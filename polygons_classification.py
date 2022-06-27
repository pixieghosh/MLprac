#%%
import numpy as np
import pandas as pd
import torch 
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from torch import nn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch.utils.data as td
from scipy.stats import linregress
import torchvision as tv
import os 
from glob import glob
from PIL import Image
# %%
# allows for images to be read
images = glob('./images/content/images/*.png', recursive= True)
# %%
img = Image.open(images[0])
# %%
plt.imshow(img)
# %%
t = tv.transforms.PILToTensor()(img)
# %%
b = np.asarray(img)
b = np.transpose(b, (2,0,1))
# %%
class PolygonsDataSet(td.Dataset):
    def __init__(self, x, y):
        self.x = x 
        self.y = y 
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self,idx):
        img = Image.open('./images/content/images/' + self.x[idx])
        img = tv.transforms.PILToTensor()(img)
        return (img, self.y[idx] - 3)
# %%
data = pd.read_csv('targets.csv')
x_train,x_test,y_train,y_test = train_test_split(data['filename'], data['sides'], test_size= 0.2, shuffle= True, stratify= data['sides'], random_state= 777)
x_train,x_val,y_train,y_val = train_test_split(x_train, y_train, test_size= 0.2, shuffle= True, stratify= y_train, random_state= 777)

# %%
traindata = PolygonsDataSet(x_train.values, y_train.values)
train_data_loader = td.DataLoader(traindata,batch_size= 32, shuffle = True)
for idx, (x_batch, y_batch) in enumerate(train_data_loader):
    pass

# %%
