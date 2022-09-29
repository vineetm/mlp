from intents import get_intents_and_labels, build_rev_index

import numpy as np


np.random.seed(42)

def create_kfold_data(sentences, labels, k=5):
  n = len(sentences)
  num_test = n // k
  num_train = n - num_test
  print(f'#Train:{num_train} #Test:{num_test}')

  k_folds = []
  for _ in range(k):
    indexes = np.random.permutation(n)
   
    k_fold = {
      'train': indexes[num_test:],
      'test': indexes[:num_test]
    }
    k_folds.append(k_fold)
  return k_folds

sentences, labels, intent_index = get_intents_and_labels('ws_2022-09-23.json')
rev_index = build_rev_index(intent_index)

k_folds = create_kfold_data(sentences, labels, 5)
assert len(k_folds) == 5
for i in range(5):
  assert len(k_folds[i]['train']) + len(k_folds[i]['test']) == len(sentences)
  
    
    

