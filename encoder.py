from sentence_transformers import SentenceTransformer
import numpy as np
import tensorflow_hub as hub
import tensorflow_text


USE_MULTI_URL = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'

class USEEncoder:
  def __init__(self, model_url=USE_MULTI_URL):
    self.embed = hub.load(model_url)
    vec = self.embed(["this is a test"])
    unit_norm = False
    if np.isclose(np.linalg.norm(vec), 1.0):
      unit_norm = True
    self.dim, self.unit_norm = vec.shape[1], unit_norm

  def encode(self, sentences):
    return self.embed(sentences).numpy()

class SentenceEncoder:
  def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    self.model_name = model_name
    self.encoder = SentenceTransformer(self.model_name)

    vec = self.encoder.encode(['test sentence'], convert_to_tensor=True)

    unit_norm = False
    if np.isclose(np.linalg.norm(vec), 1.0):
      unit_norm = True

    self.dim, self.unit_norm = vec.shape[1], unit_norm

  def encode(self, sentences, np=True):
    if np:
      return self.encoder.encode(sentences, convert_to_numpy=True)
    else:
      return self.encoder.encode(sentences, convert_to_tensor=True)
  
encoder = SentenceEncoder()
print(encoder.dim, encoder.unit_norm)