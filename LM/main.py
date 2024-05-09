import os
import torch
from utils import *
from functions import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import math
import argparse
import numpy as np
from model import LM_LSTM

def get_model():
    pass

def get_data(batch_train=20, batch_dev=512, batch_test=512):
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
    train_loader = DataLoader(train_dataset, batch_size=batch_train, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_dev, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=batch_test, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    return lang, vocab, train_loader, dev_loader, test_loader

def main(device='cuda:0', task='11', model=None):
    lang, vocab, train_loader, dev_loader, test_loader = get_data(batch_train=64)

    model_params = {
        'emb_size': 500,
        'hidden_size': 500,
        'output_size': len(lang.word2id),
        'pad_index': lang.word2id["<pad>"],
        'out_dropout': 0.5,
        'emb_dropout': 0.5,
        'n_layers': 1,
        'weight_tying': True,
        'variational_dropout': True
    }

    model = LM_LSTM(**model_params).to(device)

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    lr = 10
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
    clip = 5

    n_epochs = 100
    patience = 5
    k = 0
    t = 0
    monotone = 5
    logs = []
    best_model = None
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_loss = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    
    try:
        for epoch in pbar:
            loss_train = train_loop(train_loader, optimizer, criterion_train, model, clip=clip)
            ppl, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            scheduler.step()
            sampled_epochs.append(epoch)
            losses_train.append(loss_train)
            losses_dev.append(loss_dev)
            logs.append(ppl)
            pbar.set_postfix({'loss_train': loss_train, 'loss_dev': loss_dev, 'ppl': ppl})
            if loss_dev < best_loss:
                best_loss = loss_dev
                best_model = copy.deepcopy(model)
                patience = 5
            else:
                patience -= 1
            if patience <= 0:
                break
    except KeyboardInterrupt:
        print("Training interrupted")
    
    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)

    plot_model(sampled_epochs, losses_train, losses_dev, f'{task}/{int(final_ppl)}.png')
    save_model(best_model, optimizer, f'bin/{task}/{int(final_ppl)}.pt')


if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != 'LM':
        os.chdir('/home/disi/NLU/LM')
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    argparser.add_argument('--task', type=str, default='23', help='Task to run')
    argparser.add_argument('--model', type=str, default='', help='Model to use')
    args = argparser.parse_args()
    
    main(args.device, args.task, args.model)