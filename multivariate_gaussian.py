from operator import mul
from scipy.stats import multivariate_normal
import numpy as np
import logging
from encoder import SentenceEncoder
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

sentences, labels, intent_index = get_intents_and_labels('ws_2022-09-23.json')
labels = np.array(labels)
rev_index = build_rev_index(intent_index)

k_folds = create_kfold_data(sentences, labels)
train, heldout = k_folds[0]['train'][:-500], k_folds[0]['train'][-500:]
print(sentences[train[0]], rev_index[labels[train[0]]])
print(sentences[heldout[0]], rev_index[labels[heldout[0]]])

encoder = SentenceEncoder()
vecs = encoder.encode(sentences)

mu, log_pi, sigma = fit_multivariate_gaussian(vecs[train], labels[train], len(intent_index), c=0.0001)
preds = predict_labels(vecs[heldout], labels[heldout], len(intent_index), mu, log_pi, sigma)
print(preds, len(heldout), preds/len(heldout))
