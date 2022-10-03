import torch
import logging
import numpy as np
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
  def __init__(self, num_classes):
    super().__init__()

    self.layers = nn.Sequential(
      nn.ReLU(),
      nn.Linear(384, 128),
      nn.ReLU(),
      nn.Linear(128, num_classes)
    )
    self.ce = nn.CrossEntropyLoss()

  def forward(self, x):
    return self.layers(x)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=1e-4)

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
vecs = encoder.encode(sentences)
train_ds = IntentsDataset(vecs, labels, k_folds[0]['train'][:-500])
val_ds = IntentsDataset(vecs, labels, k_folds[0]['train'][-500:])
test_ds = IntentsDataset(vecs, labels, k_folds[0]['test'])

trainer = pl.Trainer(deterministic=True, max_epochs=500, callbacks=[EarlyStopping(monitor='val_avg_acc', patience=20, mode='max')],)
# mlp = MLP(num_classes=len(intent_index))
# trainer.validate(mlp, DataLoader(val_ds, batch_size=10))
# trainer.fit(mlp, DataLoader(train_ds, batch_size=10), DataLoader(val_ds, batch_size=10))
# trainer.test(mlp, DataLoader(test_ds, batch_size=10))

mlp =  MLP.load_from_checkpoint('./lightning_logs/version_0/checkpoints/epoch=277-step=86458.ckpt', num_classes=len(intent_index))
trainer.validate(mlp, DataLoader(val_ds, batch_size=10))
trainer.validate(mlp, DataLoader(test_ds, batch_size=10))
