import logging
import numpy as np
from scipy.stats import multivariate_normal

class MultiVariateGaussian:
    
  def fit(self, x, y, num_labels, c):
    d = x.shape[1]
    mu, log_pi, sigma = np.zeros((num_labels, d)), np.zeros(num_labels), np.zeros((num_labels, d, d))

    for label in range(num_labels):
      log_pi[label] = np.log(np.sum(y==label)/len(y))
      mu[label] = np.mean(x[y==label, :], axis=0)
      sigma[label] = np.cov(x[y==label, :], rowvar=False, bias=True) + (c * np.eye(d, d))

    return mu, log_pi, sigma

  def predict(self, testx, mu, log_pi, sigma):
    num_labels = len(mu)
    scores = np.zeros((num_labels, len(testx)))

    for label in range(num_labels):
      scores[label] = log_pi[label] + multivariate_normal.logpdf(testx, mu[label], sigma[label])
    
    return np.argmax(scores, axis=0)

  def finetune_c(self, vecs, labels, k_folds, num_labels, fraction_heldout=0.3, start_c=1e-4, max_c=1, factor=10):
    num_heldout = int(len(k_folds[0]['train']) * fraction_heldout)
    num_train = len(k_folds[0]['train']) - num_heldout
    logging.info(f'Fitting C. #Train:{num_train} #Heldout{num_heldout}')

    best_acc, best_c = 0.0, None

    c = start_c
    while c < max_c:
      acc = np.zeros(len(k_folds))
      for i, k_fold in enumerate(k_folds):
        train, heldout = k_fold['train'][:num_train], k_fold['train'][num_train:]
        logging.info(f'{vecs.shape} train_vecs shape: {train}')

        mu, log_pi, sigma = self.fit(vecs[train], labels[train], num_labels, c)
        try:
          preds = self.predict(vecs[heldout], mu, log_pi, sigma)
        except:
          logging.info(f'{c} exception')
        if np.any(preds):
          preds == len(heldout)
          num_correct = np.sum(preds == labels[heldout])
          logging.info(f'{c} Fold#:{i} #Correct:{num_correct} Acc: {num_correct/len(heldout):0.3f}')
          acc[i] = num_correct/len(heldout)
        
        avg_acc = np.average(acc)
        if avg_acc > best_acc:
          best_c, best_acc = c, avg_acc
      c *= factor
    return best_c, best_acc