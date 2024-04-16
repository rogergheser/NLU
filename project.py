import torch
from utils import *
from functions import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import math
import argparse
import numpy as np

def get_model(task, vocab_len, pad_index):
    from model import LM_LSTM
    if task == '11':
        return LM_LSTM(emb_size=250, hidden_size=300, output_size=vocab_len, pad_index=pad_index, out_dropout=0, emb_dropout=0, n_layers=1, weight_tying=False, variational_dropout=False)
    elif task == '12':
        return LM_LSTM(emb_size=250, hidden_size=300, output_size=vocab_len, pad_index=pad_index, out_dropout=.2, emb_dropout=.65, n_layers=1, weight_tying=False, variational_dropout=False)
    elif task == '13':
        return LM_LSTM(emb_size=250, hidden_size=300, output_size=vocab_len, pad_index=pad_index, out_dropout=.2, emb_dropout=.65, n_layers=1, weight_tying=False, variational_dropout=False)
    elif task == '21':
        return LM_LSTM(emb_size=300, hidden_size=300, output_size=vocab_len, pad_index=pad_index, out_dropout=.2, emb_dropout=.65, n_layers=1, weight_tying=True, variational_dropout=True)
    elif task == '22':
        return LM_LSTM(emb_size=600, hidden_size=600, output_size=vocab_len, pad_index=pad_index, out_dropout=.3, emb_dropout=.65, n_layers=1, weight_tying=True, variational_dropout=True)
    elif task == '23':
        return LM_LSTM(emb_size=600, hidden_size=600, output_size=vocab_len, pad_index=pad_index, out_dropout=.3, emb_dropout=.65, n_layers=2, weight_tying=True, variational_dropout=True)
    else:
        raise ValueError("Model not found")


def main(task, device='cuda:0'):

    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    vocab = get_vocab(train_raw, special_tokens=["<pad>", "<eos>"])
    assert len(vocab) == 10001, "Vocab should have 10001 tokens"

    lang = Lang(train_raw, special_tokens=["<pad>", "<eos>"])

    # Datasets:
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Dataloaders:
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    # Set task specific parameters
    n_epochs = 60
    patience = 3

    model = get_model(task, len(vocab), lang.word2id["<pad>"]).to(device)

    if task == '11':
        clip = 5
        lr = 1
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif task == '12':
        clip = 5
        lr = 1
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif task == '13':
        clip = 5
        lr = 0.001
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif task == '21':
        clip = 5
        lr = .001
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif task == '22':
        clip = 5
        lr = .003
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif task == '23':
        clip = 5
        lr = 3
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        k = 0
        t = 0
        T = 0
        L = 5 # number of iterations in epoch
        monotone = 5
        logs = []
    else:
        raise ValueError("Task not found")

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')


    # Training loop
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    stored_loss = float('inf')
    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
        
        if epoch % 1 == 0:
            if task == '23':
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                logs.append(ppl_dev)
                if loss_dev < stored_loss:
                    stored_loss = loss_dev
                    ###
                if len(logs) > monotone and loss_dev > min(logs[:-monotone-1]):
                    print("Using ASGD")
                    optimizer = torch.optim.ASGD(model.parameters(), lr=lr, t0=0, lambd=0., weight_decay=0)
                
                #### paper implements weight decay here
            else:
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
            
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1

            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    argparser.add_argument('--task', type=str, default='22', help='Task to run')
    args = argparser.parse_args()
    
    main(args.task, args.device)