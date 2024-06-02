import torch
import os
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import regex as re
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from functions import IntentsAndSlots, init_weights, train_loop, eval_loop, collate_fn
from utils import Lang, get_splits, load_data, save_model, plot_model
from model import ModelIAS

PAD_TOKEN = 0

def main(train_loader, dev_loader, test_loader, 
        lang, model, optimizer, criterion_slots,
        criterion_intents, clip=5, device='cpu'):
    
    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient


    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    n_epochs = 100
    runs = 5

    slot_f1s, intent_acc = [], []
    for x in tqdm(range(0, runs)):
        model = ModelIAS(hid_size, out_slot, out_int, emb_size, 
                        vocab_len, pad_index=PAD_TOKEN).to(device)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()

        patience = 3
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        for x in range(1,n_epochs):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            criterion_intents, model)
            if x % 5 == 0:
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                            criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['total']['f']

                if f1 > best_f1:
                    best_f1 = f1
                else:
                    patience -= 1
                if patience <= 0: # Early stopping with patient
                    break

        plot_model(losses_train, losses_dev, dir='plots/')
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                                criterion_intents, model, lang)
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    with open('results.txt', 'w') as f:
        f.write('LR ' + str(lr) + '\n')
        f.write('Hidden Size ' + str(hid_size) + '\n')
        f.write('Embedding Size ' + str(emb_size) + '\n')
        f.write('Optimizer ' + 'Adam' + '\n')
        f.write('Slot F1 ' + str(round(slot_f1s.mean(),3)) + ' +- ' + str(round(slot_f1s.std(),3)) + '\n')
        f.write('Intent Acc ' + str(round(intent_acc.mean(), 3)) + ' +- ' + str(round(slot_f1s.std(), 3)) + '\n')
    print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))

    idx = [n for n in os.listdir('bin/')]
    if len(idx) == 0:
        idx = 0
    else:
        idx = max([int(re.findall(r'\d+', x)[0]) for x in idx]) + 1
    
    save_model(model, optimizer, f'bin/{idx}_model.pt')
    
if __name__ == '__main__':
    # TODO Test script on cuda
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_data = load_data('dataset/ATIS/train.json')
    test_data = load_data('dataset/ATIS/test.json')
    train_raw, dev_raw, test_raw, y_train, y_dev, y_test = get_splits(train_data, test_data)

    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=0)

    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=collate_fn)


    hid_size = 400
    emb_size = 500

    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token
    main(train_loader, dev_loader, test_loader,
         lang, model, optimizer, criterion_slots,
         criterion_intents, clip=clip, device=device)