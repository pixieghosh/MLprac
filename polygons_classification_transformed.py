#%%
from math import dist
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
from tqdm.notebook import tqdm

# %%
class PolygonsDataSetTransformed(td.Dataset):
    def __init__(self, x, y):
        self.x = x 
        self.y = y 
        self.transforms = [tv.transforms.Grayscale(num_output_channels= 3),
        tv.transforms.RandomHorizontalFlip(p = 1),
        tv.transforms.RandomVerticalFlip(p = 1),
        tv.transforms.RandomPerspective(p = 1, distortion_scale= 0.25),
        tv.transforms.ColorJitter(brightness= 0.4, contrast= 0.3, saturation= 0.4, hue= 0.3),
        tv.transforms.RandomAffine(degrees= 15, translate= (0.3,0.5),scale = (0.4,0.4), shear = 0.2),
        tv.transforms.RandomInvert(p = 1),
        tv.transforms.RandomSolarize(threshold= 128, p = 1),
        tv.transforms.PILToTensor()]
        self.current_transform = 0
    def __len__(self):
        return self.x.shape[0]*len(self.transforms)
    def __getitem__(self,idx):
        if idx < self.x.shape[0]:
            img = Image.open('./images/content/images/' + self.x.iloc[idx])
            sides = self.y.iloc[idx] - 3
        else:
            img = Image.open('./images/content/images/' + self.x.iloc[idx%self.x.shape[0]])
            sides =  self.y.iloc[idx%self.y.shape[0]] - 3
        img = self.transforms[self.current_transform](img)
        if idx%self.x.shape[0] == 0:
            self.current_transform += 1
        return (tv.transforms.PILToTensor()(img), sides)
    
class PolygonsDataSet(td.Dataset):
    def __init__(self, x, y):
        self.x = x 
        self.y = y 
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self,idx):
        img = Image.open('./images/content/images/' + self.x.iloc[idx])
        img = tv.transforms.PILToTensor()(img)
        return (img, self.y.iloc[idx] - 3)

class PolygonsNet(nn.Module):
    def __init__(self):
        super().__init__()
        hidden1 = nn.Conv2d(in_channels = 3, out_channels= 8, kernel_size= 3)
        relu1 = nn.ReLU()
        hidden2 = nn.Conv2d(in_channels = 8, out_channels= 16, kernel_size= 3)
        relu2 = nn.ReLU()
        hidden3 = nn.Conv2d(in_channels = 16, out_channels= 32, kernel_size= 3)
        relu3 = nn.ReLU()
        pool = nn.MaxPool2d(kernel_size= 3)
        flat = nn.Flatten()
        hidden4 = nn.Linear(32*40*40,2048)
        relu4 = nn.ReLU()
        hidden5 = nn.Linear(2048,1024)
        relu5 = nn.ReLU()
        hidden6 = nn.Linear(1024,4)
        softmax = nn.Softmax(dim = 1)
        self.model = nn.Sequential(
        hidden1,
        relu1,
        hidden2,
        relu2,
        hidden3,
        relu3,
        pool,
        flat,
        hidden4,
        relu4,
        hidden5,
        relu5,
        hidden6, 
        softmax
        )
        
    def forward(self,x):
        return self.model(x)


# %%
data = pd.read_csv('targets.csv')
x_train,x_test,y_train,y_test = train_test_split(data['filename'], data['sides'], test_size= 0.2, shuffle= True, stratify= data['sides'], random_state= 777)
x_train,x_val,y_train,y_val = train_test_split(x_train, y_train, test_size= 0.2, shuffle= True, stratify= y_train, random_state= 777)

# %%
# how to find shape for hidden4
t = torch.randn((1,3,128,128))
model = PolygonsNet()
model.forward(t).shape

#%%
#preparing for training  
epochs = 200 
batch_size = 32 
learning_rate = 0.001 
loss_fn = nn.CrossEntropyLoss()
model = PolygonsNet()
model = model.to(device = 'cuda')
optim = torch.optim.SGD(model.parameters(),lr = learning_rate)
traindata = PolygonsDataSetTransformed(x_train, y_train)
train_data_loader = td.DataLoader(traindata,batch_size= batch_size, shuffle = True)
valdata = PolygonsDataSet(x_val, y_val)
val_data_loader = td.DataLoader(valdata,batch_size= batch_size, shuffle = True)

# %%
epoc_hist = {'train_acc':[],'train_loss':[],'val_acc': [], 'val_loss': []}
for epoch in range(0,epochs): #epoch loop
    batch_acc = []
    batch_loss = []
    model.train()
    train_itr = tqdm(enumerate(train_data_loader), desc = f't epoch {epoch}', total  = len(train_data_loader), colour = 'purple')
    for idx, (x_batch, y_batch) in train_itr: # training loop per epoch
        probs = model.forward(x_batch.to(dtype = torch.float32, device = 'cuda'))
        loss = loss_fn(probs, y_batch.reshape(-1).to(device = 'cuda'))
        optim.zero_grad() # resets the optimizer
        loss.backward() # doing first derivative (calc gradients)
        optim.step() # adjusting weights
        batch_loss.append(loss.item())
        batch_acc.append(accuracy_score(y_batch.cpu().detach().numpy(),np.argmax(probs.cpu().detach().numpy(), axis = 1)))
        train_itr.set_postfix_str(s=f'{batch_loss[-1]}/{batch_acc[-1]}', refresh=True)
    epoc_hist['train_acc'].append(np.average(batch_acc))  
    epoc_hist['train_loss'].append(np.average(batch_loss))
    batch_acc = []
    batch_loss = []
    model.eval()
    val_iter = tqdm(enumerate(val_data_loader), desc = f'v epoch {epoch}', total  = len(val_data_loader), colour = 'pink')
    for idx, (x_batch, y_batch) in val_iter: # val loop per epoch
        probs = model.forward(x_batch.to(dtype = torch.float32, device = 'cuda'))
        loss = loss_fn(probs, y_batch.reshape(-1).to(device = 'cuda'))
        batch_loss.append(loss.item())
        batch_acc.append(accuracy_score(y_batch.cpu().detach().numpy(),np.argmax(probs.cpu().detach().numpy(), axis = 1)))
        val_iter.set_postfix_str(s=f'{batch_loss[-1]}/{batch_acc[-1]}', refresh=True)
    epoc_hist['val_acc'].append(np.average(batch_acc))  
    epoc_hist['val_loss'].append(np.average(batch_loss))
    print(f'training loss: {epoc_hist["train_loss"][-1]}, accuracy: {epoc_hist["train_acc"][-1]}')
    print(f'validation loss: {epoc_hist["val_loss"][-1]}, accuracy: {epoc_hist["val_acc"][-1]}')

#%%
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (10,8))
sns.lineplot(x = range(1,epochs + 1), y = epoc_hist['train_acc'], ax = ax1)
sns.lineplot(x = range(1,epochs + 1), y = epoc_hist['val_acc'], ax = ax1)
sns.lineplot(x = range(1,epochs + 1), y = epoc_hist['train_loss'], ax = ax2)
sns.lineplot(x = range(1,epochs + 1), y = epoc_hist['val_loss'], ax = ax2)


#%%
img = Image.open("D:/work/MLprac/images/content/images/000cf421-6725-4dee-bf37-04525ba04340.png")
vert_flip = tv.transforms.RandomVerticalFlip(p = 1)
vert_img = vert_flip(img)


# %%
plt.imshow(img)
# %%
plt.imshow(vert_img)
# %%
