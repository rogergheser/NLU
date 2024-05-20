import json
import os
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
import torch.backends
from torch.utils import data

PAD_TOKEN = 0
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

def get_splits(train_data, test_data, train_ratio=0.1):
    '''
        input: train_data, train_ratio
        output: train_set, dev_set
    '''
    train_set = []
    dev_set = []
    intents = [x['intent'] for x in train_data] # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(train_data[id_y])
            labels.append(y)
        else:
            mini_train.append(train_data[id_y])
    
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=train_ratio, 
                                                    random_state=42, 
                                                    shuffle=True,
                                                    stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev
    
    y_test = [x['intent'] for x in test_data]
    return train_raw, dev_raw, test_data, y_train, y_dev, y_test

class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
