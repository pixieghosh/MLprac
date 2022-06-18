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


#%%
train = pd.read_csv('fashion-mnist_train.csv')
x_train,x_val,y_train,y_val = train_test_split(train.drop(['label'], axis = 1), train['label'], test_size= 0.2, shuffle= True, stratify= train['label'], random_state= 777)


'''
#how to show pics 
for i in range(0,3):
    pic = train.iloc[i:i+1,1:].values
    pic = pic.reshape(28,28)
    plt.imshow(pic)
    plt.show()
'''
# %%
class FashionNet(nn.Module):
    def __init__(self):
        super().__init__()
        hidden1 = nn.Linear(28*28,300)
        relu1 = nn.ReLU()
        hidden2 = nn.Linear(300,300)
        relu2 = nn.ReLU()
        hidden3 = nn.Linear(300,10)
        softmax = nn.Softmax(dim = 1)
        self.model = nn.Sequential(hidden1,relu1,hidden2,relu2,hidden3,softmax)
    def forward(self,x):
        return self.model(x)  
#%%
class FashionDataSet(td.Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    def __len__(self):
        return self.x_train.shape[0]
    def __getitem__(self,idx):
        return (self.x_train.iloc[idx:idx+1,:].values.ravel(), self.y_train.iloc[idx:idx+1].values)

# %%
epochs = 180
batch_size = 32 
traindata = FashionDataSet(x_train, y_train)
train_data_loader = td.DataLoader(traindata,batch_size= batch_size, shuffle = True)
batch_count = len(train_data_loader)
learning_rate = 0.005
loss_fn = nn.CrossEntropyLoss() 
model = FashionNet()
model = model.to(device = 'cuda')
optim = torch.optim.SGD(model.parameters(),lr = learning_rate)

#%%
epoc_hist = {'train_acc':[],'train_loss':[],'val_acc': [], 'val_loss': []}
for epoch in range(0,epochs): #epoch loop
    batch_acc = []
    batch_loss = []
    for idx, (x_batch, y_batch) in enumerate(train_data_loader): # batch loop
        probs = model.forward(x_batch.to(dtype = torch.float32, device = 'cuda'))
        loss = loss_fn(probs, y_batch.reshape(-1).to(device = 'cuda'))
        optim.zero_grad() # resets the optimizer
        loss.backward() # doing first derivative (calc gradients)
        optim.step() # adjusting weights
        batch_loss.append(loss.item())
        batch_acc.append(accuracy_score(y_batch.cpu().detach().numpy(),np.argmax(probs.cpu().detach().numpy(), axis = 1)))
        if idx%100 == 0:
            print(f'batch {idx} loss: {batch_loss[-1]}, accuracy: {batch_acc[-1]}')
    epoc_hist['train_acc'].append(np.average(batch_acc))  
    epoc_hist['train_loss'].append(np.average(batch_loss))
    probs_val = model.forward(torch.from_numpy(x_val.values).to(dtype=torch.float32, device = 'cuda'))
    loss_val = loss_fn(probs_val, torch.from_numpy(y_val.values.reshape(-1)).to(device = 'cuda'))  
    epoc_hist['val_acc'].append(accuracy_score(y_batch.cpu().detach().numpy(),np.argmax(probs.cpu().detach().numpy(), axis = 1)))
    epoc_hist['val_loss'].append(loss_val.item())
    print(f'training loss: {epoc_hist["train_loss"][-1]}, accuracy: {epoc_hist["train_acc"][-1]}')
    print(f'validation loss: {epoc_hist["val_loss"][-1]}, accuracy: {epoc_hist["val_acc"][-1]}')

# %%
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (10,8))
sns.lineplot(x = range(1,epochs + 1), y = epoc_hist['train_acc'], ax = ax1)
sns.lineplot(x = range(1,epochs + 1), y = epoc_hist['val_acc'], ax = ax1)
sns.lineplot(x = range(1,epochs + 1), y = epoc_hist['train_loss'], ax = ax2)
sns.lineplot(x = range(1,epochs + 1), y = epoc_hist['val_loss'], ax = ax2)
# %%
