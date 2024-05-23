from transformers import BertModel, BertTokenizer
import torch.nn as nn


class SlotBert(BertModel):
    def __init__(self, config, num_labels, dropout=0.1):
        super(SlotBert, self).__init__(config)
        self.bert = BertModel(config)
        self.slot_classifier = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)
        return slot_logits
