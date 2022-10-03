from scipy.stats import multivariate_normal
import numpy as np
import logging
from encoder import SentenceEncoder, USEEncoder
from intents import get_intents_and_labels, build_rev_index
from utils import create_kfold_data

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def fit_multivariate_gaussian(x, y, num_labels, c):
  d = x.shape[1]
  mu = np.zeros((num_labels, d))
  log_pi = np.zeros(num_labels)
  sigma = np.zeros((num_labels, d, d))

  for label in range(num_labels):
    log_pi[label] = np.log(np.sum(y==label)/len(y))
    mu[label] = np.mean(x[y==label, :], axis=0)
    sigma[label] = np.cov(x[y==label, :], rowvar=False, bias=True) + (c * np.eye(d, d))
  
  return mu, log_pi, sigma

def predict_labels(testx, testy, num_labels, mu, log_pi, sigma):
  scores = np.zeros((num_labels, len(testx)))

  for label in range(num_labels):
    scores[label] = log_pi[label] + multivariate_normal.logpdf(testx, mu[label], sigma[label])
  
  preds = np.argmax(scores, axis=0)
  return np.sum(preds == testy)

def fit_predict(num_labels, vecs, labels, train_indexes, test_indexes, c):
  mu, log_pi, sigma = fit_multivariate_gaussian(vecs[train_indexes], labels[train_indexes], num_labels, c=c)
  try:
    preds = predict_labels(vecs[test_indexes], labels[test_indexes], num_labels, mu, log_pi, sigma)
  except Exception:
    logging.info('exception')
    return None
  return preds

def find_c(v1, v2, vecs, labels, k_folds, num_labels, fraction_heldout=0.3):
  if v1 < v2:
    start, end = v1, v2
  else:
    start, end = v2, v1

  num_heldout = int(len(k_folds[0]['train']) * fraction_heldout)
  num_train = len(k_folds[0]['train']) - num_heldout

  best_acc, best_c = 0., None
  while not np.isclose(start, end):
    acc_start = np.zeros(len(k_folds))
    acc_end = np.zeros(len(k_folds))
    acc_mid = np.zeros(len(k_folds))

    mid = (start + end) / 2
    
    for i, k_fold in enumerate(k_folds):
      train, heldout = k_fold['train'][:num_train], k_fold['train'][num_train:]
      preds_end = fit_predict(num_labels, vecs, labels, train, heldout, end)
      preds_start = fit_predict(num_labels, vecs, labels, train, heldout, start)
      acc_start[i] = preds_start/len(heldout)
      acc_end[i] = preds_end/len(heldout)

      if np.isclose(mid, end) or np.isclose(start, mid):
        continue

      preds_mid = fit_predict(num_labels, vecs, labels, train, heldout, mid)
      acc_mid[i] = preds_mid/len(heldout)

    avg_start = np.average(acc_start)
    avg_mid = np.average(acc_mid)
    avg_end = np.average(acc_end)

    best_acc = max(avg_start, best_acc)
    best_acc = max(avg_mid, best_acc)
    best_acc = max(avg_end, best_acc)

    if np.isclose(best_acc, avg_start):
      best_c = start
    elif np.isclose(best_acc, avg_mid):
      best_c = mid
    else:
      best_c = end

    logging.info(f'{start} {end} {mid} {best_acc:0.3f}')
    if avg_mid > avg_start:
      start = mid
    else:
      end = mid
    
  return best_c, best_acc


def find_range_c(vecs, labels, k_folds, num_labels, fraction_heldout=0.3):
  num_heldout = int(len(k_folds[0]['train']) * fraction_heldout)
  num_train = len(k_folds[0]['train']) - num_heldout
  logging.info(f'Fitting C. #Train:{num_train} #Heldout{num_heldout}')

  last_c, best_c, best_acc = 0, 0, 0.
  c = 1e-5
  while c < 1:
    acc = np.zeros(len(k_folds))
    for i, k_fold in enumerate(k_folds):
      train, heldout = k_fold['train'][:num_train], k_fold['train'][num_train:]
      preds = fit_predict(num_labels, vecs, labels, train, heldout, c)
      if not preds:
        acc[i] = 0.
      else:
        acc[i] = preds/len(heldout)
    
    avg_acc = np.average(acc)
    logging.info(f'{c} {acc}')
    if avg_acc > best_acc:
      last_c = best_c
      best_c = c
      best_acc = avg_acc
    c *= 10
  return last_c, best_c, best_acc

sentences, labels, intent_index = get_intents_and_labels('ws_2022-09-23.json')
labels = np.array(labels)
rev_index = build_rev_index(intent_index)

k_folds = create_kfold_data(sentences, labels)
# encoder = SentenceEncoder()
encoder = USEEncoder()
vecs = encoder.encode(sentences)

start_c, end_c, best_acc = find_range_c(vecs, labels, k_folds, len(intent_index))
# logging.info(f'{start_c} {end_c} {best_acc: 0.3f}')

# best_c, best_acc = find_c(start_c, end_c, vecs, labels, k_folds, len(intent_index))
# logging.info(f'Range C: {best_c} {best_acc}' )

acc = np.zeros(len(k_folds))
for i in range(len(k_folds)):
  train, test = k_folds[i]['train'], k_folds[i]['test']
  preds = fit_predict(len(intent_index), vecs, labels, train, test, end_c)
  acc[i] = preds/len(test)
logging.info(f'Acc: {acc} {np.average(acc):0.3f}')
