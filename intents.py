import json

def get_intents_and_labels(ws_json):
  intent_index = {}
  with open(ws_json) as fr:
    ws = json.load(fr)
  
  sentences = []
  labels = []

  for intent in ws['intents']:
    intent_name = intent['intent']
    for example in intent['examples']:
      sentences.append(example['text'])
      if intent_name not in intent_index:
        intent_index[intent_name] = len(intent_index)
      labels.append(intent_index[intent_name])
  return sentences, labels, intent_index

def build_rev_index(intent_index):
  return {
    intent_index[intent]: intent
    for intent in intent_index
  }

sentences, labels, intent_index = get_intents_and_labels('ws_2022-09-23.json')
rev_index = build_rev_index(intent_index)