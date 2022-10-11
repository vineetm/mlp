import numpy as np
from encoder import SentenceEncoder

from intents import get_intents_and_labels, build_rev_index
from utils import create_kfold_data
from mvg import MultiVariateGaussian
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

encoder = SentenceEncoder()
sentences, labels, intent_index = get_intents_and_labels('ws_2022-09-23.json')
labels = np.array(labels)
rev_index = build_rev_index(intent_index)

k_folds = create_kfold_data(sentences, labels)
encoder = SentenceEncoder()
# encoder = USEEncoder()
vecs = encoder.encode(sentences)
logging.info(f'Vecs shape: {vecs.shape}')

mvg = MultiVariateGaussian()
# best_c, _ = mvg.finetune_c(vecs, labels, k_folds, len(intent_index))

best_c = 0.001
kfolds_acc = np.zeros(len(k_folds))
for i, kfold in enumerate(k_folds):
  mu, log_pi, sigma = mvg.fit(vecs[kfold['train']], labels[kfold['train']], len(intent_index), best_c)
  preds = mvg.predict(vecs[kfold['test']], mu, log_pi, sigma)
  assert len(preds) == len(kfold['test'])
  num_correct = np.sum(preds == labels[kfold['test']])
  kfolds_acc[i] = num_correct / len(labels[kfold['test']])

logging.info(f'{np.average(kfolds_acc): 0.3f}')