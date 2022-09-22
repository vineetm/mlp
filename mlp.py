"""
Source: https://github.com/christianversloot/machine-learning-articles/blob/main/creating-a-multilayer-perceptron-with-pytorch-and-lightning.md
"""

import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(32 * 32 * 3, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 10)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)

def main():
  torch.manual_seed(42)

  dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())


  #Train loader will created batches. Each batch would have two components x, y
  #x: 3x32x32
  #y: labels
  trainloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=1)

  mlp = MLP()
  
  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

  for epoch in range(0, 5):
    print(f'Starting {epoch}')

    current_loss = 0.0

    for i, data in enumerate(trainloader):

      x, y = data

      optimizer.zero_grad()

      outputs = mlp(x)

      loss = loss_function(outputs, y)

      loss.backward()

      optimizer.step()

      current_loss += loss.item()

      if i%500 == 499:
        print(f'Loss after mini-batch {i+1:5d} {current_loss/500: 0.3f}')
        current_loss = 0.0



if __name__ == '__main__':
  main()