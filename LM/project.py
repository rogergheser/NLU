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
        return LM_LSTM(emb_size=800, hidden_size=800, output_size=vocab_len, pad_index=pad_index, out_dropout=.3, emb_dropout=.65, n_layers=1, weight_tying=True, variational_dropout=True)
    elif task == '23':
        return LM_LSTM(emb_size=800, hidden_size=800, output_size=vocab_len, pad_index=pad_index, out_dropout=.2, emb_dropout=.6, n_layers=1, weight_tying=True, variational_dropout=True)
    else:
        raise ValueError("Model not found")


def main(task, device='cuda:0', model=None):

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
    train_loader = DataLoader(train_dataset, batch_size=20, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=512, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=512, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    # Set task specific parameters
    n_epochs = 100
    patience = 7

    if model is None or model == '':
        model = get_model(task, len(lang.word2id), lang.word2id["<pad>"]).to(device)
    else:
        model = torch.load(model).to(device)

    scheduler = None
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
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    elif task == '23':
        clip = 5
        lr = 10
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
        # optimizer = torch.optimx.AdamW(model.parameters(), lr=lr2)
        k = 0
        t = 0
        T = 0
        L = 5 # number of iterations in epoch
        monotone = 5
        logs = []
    else:
        raise ValueError("Task not found")

    # Training loop
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_loss = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    stored_loss = float('inf')
    #If the PPL is too high try to change the learning rate
    changed = True
    try:
        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)            
            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                
                if 't0' in optimizer.param_groups[0]: # ASGD
                    tmp = {}
                    for prm in model.parameters():
                        tmp[prm] = prm.data.clone()
                        prm.data = optimizer.state[prm]['ax'].clone()

                    ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)

                    if ppl_dev < best_ppl:
                        best_ppl = ppl_dev
                        best_model = copy.deepcopy(model).to('cpu')
                        patience = 5
                    else:
                        patience -= 1
                    if patience <= 0:
                        break
                    for prm in model.parameters():
                        prm.data = tmp[prm].clone()
                else:
                    ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                    if 't0' not in optimizer.param_groups[0] \
                            and (len(logs)) > monotone and loss_dev > min(logs[:-monotone]):
                        print("Using ASGD")
                        optimizer = torch.optim.ASGD(model.parameters(), lr=lr, t0=0, lambd=0.)

                    logs.append(loss_dev)

                losses_dev.append(np.asarray(loss_dev).mean())
                
                pbar.set_description("PPL: {} | Loss {}".format(ppl_dev, loss_dev))
                if ppl_dev < best_ppl:
                        best_ppl = ppl_dev
                        best_model = copy.deepcopy(model).to('cpu')
                        patience = 5
                else:
                    patience -= 1
                if patience <= 0:
                    break
                
            if scheduler is not None:
                scheduler.step()   

    except KeyboardInterrupt:
        print("Training interrupted")
        pass

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)

    plot_model(sampled_epochs, losses_train, losses_dev, f'{task}/{int(final_ppl)}.png')
    save_model(best_model, optimizer, f'bin/{task}/{int(final_ppl)}.pt')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    argparser.add_argument('--task', type=str, default='23', help='Task to run')
    argparser.add_argument('--model', type=str, default='', help='Model to use')
    args = argparser.parse_args()
    
    main(args.task, args.device, args.model)