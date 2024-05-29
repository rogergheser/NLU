import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

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