"""
Source: https://github.com/christianversloot/machine-learning-articles/blob/main/creating-a-multilayer-perceptron-with-pytorch-and-lightning.md
"""

import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl


class MLP(pl.LightningModule):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(32 * 32 * 3, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 10)
    )
    self.ce = nn.CrossEntropyLoss()


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)

  def training_step(self, batch, batch_id):
    x, y = batch
    x = x.view(x.size(0), -1)
    y_hat = self.layers(x)
    loss = self.ce(y_hat, y)
    self.log('train_loss', loss)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=1e-4)

def main():
  dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
  pl.seed_everything(42)
  mlp = MLP()

  trainer = pl.Trainer(auto_scale_batch_size='power', deterministic=True, max_epochs=5)
  trainer.fit(mlp, DataLoader(dataset))
  


if __name__ == '__main__':
  main()