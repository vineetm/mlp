from re import L
from annoy import AnnoyIndex
from encoder import SentenceEncoder
from intents import get_intents_and_labels, build_rev_index
import numpy as np
from utils import create_kfold_data
import logging
import time
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

np.random.seed(12)

def train_model(vecs, indexes, encoder_dim, metric='dot', num_trees=10):
  ann = AnnoyIndex(encoder_dim, metric=metric)
  for i, index in enumerate(indexes):
    ann.add_item(i, vecs[index])
  ann.build(num_trees)
  return ann

def get_majority_label(labels, train_indexes, nns):
  if len(nns) == 1:
    return labels[train_indexes[nns[0]]]
  
  count = {}
  max_count, max_label = 0, ''
  for nn in nns:
    pred = labels[train_indexes[nn]]
    count[pred] = count.get(pred, 0)+1
    if count[pred] > max_count:
      max_count = count[pred]
      max_label = pred
  return max_label
  
def eval_model(ann, train_indexes, test_indexes, vecs, labels, num_nbrs=1):
  gold = np.array([
    labels[index]
    for index in test_indexes
  ])

  preds = []

  for i, index  in enumerate(test_indexes):
    nns = ann.get_nns_by_vector(vecs[index], n=num_nbrs)
    preds.append(get_majority_label(labels, train_indexes, nns))
  
  preds = np.array(preds)
  return np.sum(preds==gold)
  
encoder = SentenceEncoder()
logging.info(f'{encoder.dim} Normalized: {encoder.unit_norm}')
sentences, labels, intent_index = get_intents_and_labels('ws_2022-09-23.json')

start = time.time()
vecs = encoder.encode(sentences)
logging.info(f'#Sentences: {len(sentences)} #Intents: {len(intent_index)} Vecs:{vecs.shape} Time:{time.time()-start:0.2f}s')

rev_index = build_rev_index(intent_index)
K=5
k_folds = create_kfold_data(sentences, labels, K)

num_correct = np.zeros(K)

start = time.time()
for k in range(K):
  ann = train_model(vecs, k_folds[k]['train'], encoder.dim)
  num_correct[k] = eval_model(ann, k_folds[k]['train'], k_folds[k]['test'], vecs, labels)
logging.info(f'1-NN {num_correct} Avg: {np.average(num_correct)} %:{np.average(num_correct)/len(k_folds[k]["test"]):0.3f} Time:{time.time()-start:0.2f}s')

for k in range(K):
  ann = train_model(vecs, k_folds[k]['train'], encoder.dim)
  num_correct[k] = eval_model(ann, k_folds[k]['train'], k_folds[k]['test'], vecs, labels, num_nbrs=3)
logging.info(f'3-NN {num_correct} Avg: {np.average(num_correct)} %:{np.average(num_correct)/len(k_folds[k]["test"]):0.3f} Time:{time.time()-start:0.2f}s')