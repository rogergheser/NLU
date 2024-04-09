import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from functions import VariationalDropout

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0,
                 emb_dropout=0, n_layers=1, weight_tying=False, variational_dropout=False):
        super(LM_LSTM, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)                # Embedding layer -> idea is to use as dictionary, rows are the word embeddings
        
        self.drop1 = VariationalDropout(p=emb_dropout) if variational_dropout else nn.Dropout(p=emb_dropout)
        
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)                   # LSTM layer
        
        self.pad_token = pad_index
                
        self.drop2 = VariationalDropout(p=out_dropout) if variational_dropout else nn.Dropout(p=out_dropout)
        
        self.output = nn.Linear(hidden_size, output_size)
        
        if weight_tying: 
            assert hidden_size == emb_size, "Hidden size and embedding size must be equal for weight tying"
            self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop1 = self.drop1(emb)
        lstm_out, _  = self.lstm(drop1)
        drop2 = self.drop2(lstm_out)
        output = self.output(drop2).permute(0,2,1)
        return output