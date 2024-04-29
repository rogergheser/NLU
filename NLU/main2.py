from transformers import BertTokenizer, BertModel
import torch
from pprint import pprint

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer
model = BertModel.from_pretrained("bert-base-uncased") # Download the model

inputs = tokenizer(["I saw a man with a telescope", "StarLord was here",  "I didn't"], return_tensors="pt", padding=True)
pprint(inputs)

outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
pprint(outputs)