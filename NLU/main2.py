from transformers import BertTokenizer, BertModel
import torch
from pprint import pprint
import os
from  utils import *
from functions import *
from torch.utils.data import DataLoader
import torch.optim as optim

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer
model = BertModel.from_pretrained("bert-base-uncased") # Download the model

inputs = tokenizer(["I saw a man with a telescope", "StarLord was here",  "I didn't"], return_tensors="pt", padding=True)
pprint(inputs)

outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
pprint(outputs)





if __name__ == '__main__':
    if os.getcwd().split('/')[-2:] != 'NLU/NLU':
        os.chdir('/home/disi/NLU/NLU')

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_data = load_data('dataset/ATIS/train.json')
    test_data = load_data('dataset/ATIS/test.json')
    train_raw, dev_raw, test_raw, y_train, y_dev, y_test = get_splits(train_data, test_data)

    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute 
                                                                # the cutoff
    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, 
                                            # however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=0)

    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)


    lr = 0.00001

    # Change next lines
    model = "".to(device)
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token