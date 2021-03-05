# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 18:08:52 2021

@author: Tim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

class CharDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.labels = df['label']
        self.imgs = np.expand_dims(
            df.drop('label', axis = 1).astype('float32').to_numpy().reshape(
                len(self.labels), 28, 28), axis = 1)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.imgs[idx]
        label = letter2Index(self.labels.iloc[idx])
        return img, label
    def __len__(self):
        return len(self.labels)
         
def letter2Index(letter):
    alphabet = 'ACTG'
    return alphabet.find(letter)

class CharCNN(nn.Module):
    def __init__(self):
        super(CharCNN, self).__init__()
        self.input = nn.Conv2d(1, 25, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(25, 50, 5)
        self.dense = nn.Linear(50*4*4, 160)
        self.out = nn.Linear(160, 4)
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.pool(F.relu(self.input(x)))
        x = self.pool(F.relu(self.conv(x)))
        x = F.relu(self.dense(self.flatten(x)))
        x = self.out(x)
        return x
    
def train_model(model, criterion, optimizer, dataloader, cv, num_epochs = 1):
    training_losses = []
    cv_losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        running_loss = 0.0
        i = 0
        for x, y in dataloader: # Training step
            model.train()
            x = x.to(device)
            y = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2020 mini-batches
                print('[%d, %5d] loss: %.9f' %
                      (epoch + 1, i + 1, running_loss / (200 * len(x))))
                total_loss += running_loss / len(x)
                running_loss = 0.0
            i += 1
        training_losses.append(total_loss / i)
        total_loss = 0.0
        i = 0
        # Evaluation
        with torch.no_grad():
            for x, y in cv:
                x = x.to(device)
                y = y.to(device)  
                model.eval()
                outputs = model(x)
                loss = criterion(outputs, y)
                total_loss += loss.item() / len(x)
                i += 1
            cv_losses.append(total_loss / i)
    print('Finished Training')
    PATH = model.__class__.__name__ + '.pth'
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)
    with open('losses', 'wb') as f:
        pickle.dump([training_losses, cv_losses], f)
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ds = CharDataset("train.csv")
train_dataloader = DataLoader(ds, batch_size = 512, shuffle = True, pin_memory = True, num_workers = 6)
ds = CharDataset("CV.csv")
cv_dataloader = DataLoader(ds, batch_size = 512, shuffle = True, pin_memory = True, num_workers = 6)
model = CharCNN()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0003)

# Loading from previous checkpoint
#PATH = model.__class__.__name__ + '.pth'
#checkpoint = torch.load(PATH)
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model = train_model(model, criterion, optimizer, train_dataloader, cv_dataloader, num_epochs = 1)        