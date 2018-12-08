from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import pandas as pd

dataframe = pd.read_csv("irisdata.csv")
two_class = dataframe[dataframe['species'] != "setosa"]

two_class.loc[two_class["species"] == "virginica", "species"] = 0
two_class.loc[two_class["species"] == "versicolor", "species"] = 1

in_vec = two_class[["petal_length", "petal_width"]]
out_vec = two_class["species"]

import matplotlib.pyplot as plt 
plt.scatter(in_vec.values[:,0], in_vec.values[:,1], c=out_vec.values)
plt.colorbar()
plt.show()

print("\n\n")

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

num_in = 2
num_out = 1

class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()
    self.fullyconnected1 = nn.Linear(num_in,num_out)

  def forward(self, x):
    x = self.fullyconnected1(x)
    x = F.sigmoid(x)
    return x

model = Network()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1000
num_examples = two_class.shape[0]

model.train()

for epoch in range(num_epochs):
  for i in range(num_examples):
    attributes = torch.tensor(in_vec.iloc[i].values, dtype=torch.float)
    label = torch.tensor(out_vec.iloc[i], dtype=torch.float)
    optimizer.zero_grad()
    output = model(attributes)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
  if epoch % 100 == 0:
    print('Epoch: {} | Loss: {:.6f}'.format(epoch, loss.item()))


model.eval()

pred = torch.zeros(out_vec.shape)

for i in range(num_examples):
  attributes = torch.tensor(in_vec.iloc[i].values, dtype=torch.float)
  label = torch.tensor(out_vec.iloc[i], dtype=torch.float)

  pred[i] = model(attributes).round()

def one_if_same(a, b):
  if a == b:
    return 1
  else:
    return 0

print('Correct classifications: {}/{}'.format(sum(pred == torch.tensor(out_vec.values).float()),len(out_vec)))
