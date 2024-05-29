import os
import matplotlib.pyplot as plt
import regex as re
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from utils import get_index

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def collate_fn(data):
    input_ids = [item['input_ids'] for item in data]
    slot_labels = [item['slot_labels'] for item in data]
    intent_labels = [item['intent_labels'] for item in data]
    attention_mask = [item['attention_mask'] for item in data]
    token_type_ids = [item['token_type_ids'] for item in data]

    input_ids = torch.stack(input_ids, dim=0)
    slot_labels = torch.stack(slot_labels, dim=0)
    intent_labels = torch.stack(intent_labels, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    token_type_ids = torch.stack(token_type_ids, dim=0)

    return {'input_ids': input_ids, 'slot_labels': slot_labels, 'intent_labels': intent_labels, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

class IntentsAndSlots(torch.utils.data.Dataset):
    def __init__(self, data, lang, tokenizer, max_seq_len=64):
        self.data = data
        self.lang = lang
        self.tokenizer = tokenizer

        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        unk_token = tokenizer.unk_token
        pad_token_id = tokenizer.pad_token_id

        self.input_ids = []
        self.slot_labels = []
        self.intent_labels = []
        self.attention_mask = []
        self.token_type_ids = []

        for sample in data:
            tokens = []
            slot_labels = []
            utterance, slots, intent = sample['utterance'].split(" "), sample['slots'].split(" "), sample['intent']

            for word, slot in zip(utterance, slots):
                _tokens = tokenizer.tokenize(word)
                if not _tokens:
                    tokens = [unk_token]
                tokens.extend(_tokens)
                slot_labels.extend([lang.slot2id[slot]] + [pad_token_id]*(len(_tokens)-1) )
            
            if len(tokens) > max_seq_len - 2:
                print("Choose higher max_seq_len")
                tokens = tokens[:max_seq_len-2]
                slot_labels = slot_labels[:max_seq_len-2]

            tokens = [cls_token] + tokens + [sep_token]
            slot_labels = [pad_token_id] + slot_labels + [pad_token_id]
            self.token_type_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            self.attention_mask = [1] * len(input_ids)

            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)            
            slot_labels = slot_labels + ([pad_token_id] * padding_length)
            if len(self.attention_mask) < max_seq_len:
                self.attention_mask = self.attention_mask + ([0] * padding_length)
            if len(self.token_type_ids) < max_seq_len:
                self.token_type_ids = self.token_type_ids + ([0] * padding_length)

            self.input_ids.append(input_ids)
            self.slot_labels.append(slot_labels)
            self.intent_labels.append(lang.intent2id[intent])

            if len(slot_labels) != max_seq_len:
                print("Error in slot labels")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.input_ids[idx]),
                'slot_labels': torch.tensor(self.slot_labels[idx]),
                'intent_labels': torch.tensor(self.intent_labels[idx]),
                'attention_mask': torch.tensor(self.attention_mask),
                'token_type_ids': torch.tensor(self.token_type_ids)}