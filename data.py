from torch.utils.data import Dataset, DataLoader
from encoder import SentenceEncoder
from utils import get_intents_and_labels, create_kfold_data
class IntentsDataset(Dataset):
  
  def __init__(self, vecs, labels, indexes):
    self.vecs = vecs[indexes]
    self.labels = [
      labels[index]
      for index in indexes
    ]
    self.indexes = indexes
      

  def __len__(self):
    return len(self.vecs)

  def __getitem__(self, index):
    return {
      'x': self.vecs[index],
      'y': self.labels[index]
    }


encoder = SentenceEncoder()
sentences, labels, intent_index = get_intents_and_labels('ws_2022-09-23.json')

k_folds = create_kfold_data(sentences, labels)
vecs = encoder.encode(sentences)
ds = IntentsDataset(vecs, labels, k_folds[0]['train'])

all_batches = [
  batch
  for batch in DataLoader(ds, batch_size=10, shuffle=True)
]

print(all_batches[0]['x'].shape)