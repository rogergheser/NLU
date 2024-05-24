from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn


class SlotBert(BertPreTrainedModel):
    def __init__(self, config, num_labels, dropout=0.1):
        super(SlotBert, self).__init__(config)
        self.bert = BertModel(config)
        self.slot_classifier = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)
        return slot_logits
    def init_weights(self):
        self.slot_classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.slot_classifier.bias.data.zero_()