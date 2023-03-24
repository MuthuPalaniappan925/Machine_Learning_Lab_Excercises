#Importing Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

##Torch-Packages
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


##Loading the Dataset
df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")

X = df.drop(["Personal Loan", "ID"], axis = 1)
y = df["Personal Loan"]


##Data Pre-Processing
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


##Py-Torch

class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)

train_data = TrainData(torch.FloatTensor(X_train),torch.FloatTensor(y_train))
class TestData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__ (self):
        return len(self.X_data)

test_data = TestData(torch.FloatTensor(X_test))

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()

        self.layer_1 = nn.Linear(12, 12)
        self.layer_2 = nn.Linear(12, 8)
        self.layer_out = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.layer_out(x)

        return x
    
    
model = BinaryClassification()
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

def binary_acc(y_pred, y_test):
    y_pred = torch.round(torch.sigmoid(y_pred))
    acc = (y_pred == y_test).sum().float() / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

epoch = 50
for e in range(1, epoch+1):
    epoch_loss = 0
    epoch_acc = 0
    for X, y in train_loader:
        optimizer.zero_grad()

        y_pred = model(X)

        loss = criterion(y_pred, y.unsqueeze(1))
        acc = binary_acc(y_pred, y.unsqueeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()


    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
    
y_pred_torch = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_torch.append(y_pred_tag.cpu().numpy())

y_pred_torch = [a.squeeze().tolist() for a in y_pred_torch]

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_torch))
