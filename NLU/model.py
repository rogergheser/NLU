import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertPreTrainedModel, BertModel

class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)    
        self.slot_out = nn.Linear(hid_size*2, out_slot)
        self.intent_out = nn.Linear(hid_size*2, out_int)
        # Dropout layer How/Where do we apply it?
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        utt_encoded = self.dropout(utt_encoded)
        last_hidden = self.dropout(last_hidden)
        # Get the last hidden state
        last_hidden = torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim=1)
        
        # Is this another possible way to get the last hiddent state? (Why?)
        # utt_encoded.permute(1,0,2)[-1]
        
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent

def print_activations_hook(module, input, output):
    if torch.isnan(output).any() or torch.isinf(output).any():
        print(f"NaNs or Infs in the output of {module}")

class JointBert(BertPreTrainedModel):
    def __init__(self, config, out_slot, out_int, dropout=0.1, device='mps'):
        super(JointBert, self).__init__(config)
        self.bert = BertModel(config)
        self.slot_out = nn.Linear(config.hidden_size, out_slot)
        self.intent_out = nn.Linear(config.hidden_size, out_int)
        self.dropout = nn.Dropout(dropout)

        # self._init_weights_(self.slot_out)
        # self._init_weights_(self.intent_out)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden = outputs[0]
        pooled_output = outputs[1]

        if torch.isnan(last_hidden).any():
            print("Nan in last_hidden")
        for name, param in self.bert.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"NaNs or Infs found in {name}")


        last_hidden = self.dropout(last_hidden)        
        # Compute slot logits
        slots = self.slot_out(last_hidden).permute(0,2,1)
        # Compute intent logits
        intent = self.intent_out(pooled_output)
        
        return slots, intent
    
    def _init_weights_(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)