from asyncio import base_events
import os
import torch
import logging
import numpy as np
from pathlib import Path
from torch import nn
import pytorch_lightning as pl
from encoder import SentenceEncoder
from data import IntentsDataset
from torch.utils.data import DataLoader
from utils import create_kfold_data, get_intents_and_labels
from pytorch_lightning.callbacks import EarlyStopping

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MLP(pl.LightningModule):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, num_classes, d, lr=1e-4):
    super().__init__()

    self.layers = nn.Sequential(
      nn.ReLU(),
      nn.Linear(d, 128),
      nn.ReLU(),
      nn.Linear(128, num_classes)
    )
    self.ce = nn.CrossEntropyLoss()
    self.lr = lr

  def forward(self, x):
    return self.layers(x)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.lr)

  def __step(self, batch):
    x, y = batch['x'], batch['y']
    y_hat = self.layers(x)
    loss = self.ce(y_hat, y)
    return loss

  def training_step(self, batch, batch_id):
    loss = self.__step(batch)
    self.log('train_loss', loss, prog_bar=True)
    return loss

  @torch.no_grad()
  def validation_step(self, batch, batch_id):
    x, y = batch['x'], batch['y']
    y_hat = self.layers(x)
    loss = self.ce(y_hat, y)
    self.log('val_loss', loss, prog_bar=True)
    return {'preds': torch.argmax(y_hat, axis=1), 'gold': y, 'loss': loss}

  def validation_epoch_end(self, outputs):
    num_correct, total = 0, 0
    for output in outputs:
      num_correct += torch.sum(output['preds'] == output['gold'])
      total += len(output['gold'])
    acc = num_correct / total
    self.log('val_avg_acc', acc, prog_bar=True)
    return {'val_avg_acc': acc}
  
  def test_epoch_end(self, outputs):
    num_correct, total = 0, 0
    for output in outputs:
      num_correct += torch.sum(output['preds'] == output['gold'])
      total += len(output['gold'])
    acc = num_correct / total
    logging.info(f'Test Acc: {acc: 0.3f}')
    return {'test_avg_acc': acc}
    
  @torch.no_grad()
  def test_step(self, batch, batch_id):
    x, y = batch['x'], batch['y']
    y_hat = self.layers(x)
    loss = self.ce(y_hat, y)
    return {'preds': torch.argmax(y_hat, axis=1), 'gold': y, 'loss': loss}

pl.seed_everything(42)
encoder = SentenceEncoder()
sentences, labels, intent_index = get_intents_and_labels('ws_2022-09-23.json')

k_folds = create_kfold_data(sentences, labels)

n = len(k_folds[0]['train'])
num_heldout = int(n * 0.1)
num_train = n - num_heldout
logging.info(f'#Train #Heldout: {num_train} {num_heldout}')

vecs = encoder.encode(sentences)
num_labels = len(intent_index)

base_dir = Path('k-folds2')
base_dir.mkdir(exist_ok=True)

for i in range(len(k_folds)):
  logging.info(f'Fold-{i} Start')
  fold_dir = Path(base_dir) / f'fold-{i}'
  fold_dir.mkdir(exist_ok=True)

  trainer = pl.Trainer(deterministic=True, max_epochs=500, callbacks=[EarlyStopping(monitor='val_avg_acc', patience=20, mode='max')], default_root_dir=fold_dir)
  mlp = MLP(num_labels, d=encoder.dim)
  train_ds = IntentsDataset(vecs, labels, k_folds[i]['train'][:num_train])
  val_ds = IntentsDataset(vecs, labels, k_folds[i]['train'][num_train:])
  trainer.fit(mlp, DataLoader(train_ds, batch_size=10), DataLoader(val_ds, batch_size=10))

  ckpt_dir = fold_dir / 'lightning_logs/version_0/checkpoints'
  ckpt = fold_dir / 'lightning_logs/version_0/checkpoints' / os.listdir(ckpt_dir)[0]
  mlp_test = MLP.load_from_checkpoint(ckpt, num_classes=len(intent_index), d=encoder.dim)

  test_ds = IntentsDataset(vecs, labels, k_folds[i]['test'])
  trainer.test(mlp_test, DataLoader(test_ds, batch_size=10))
  logging.info(f'Fold-{i} Done')