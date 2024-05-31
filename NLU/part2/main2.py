from matplotlib import pyplot as plt
import torch
import os
import torch.optim as optim
import numpy as np
import pickle
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm
from utils import get_index
from conll import evaluate
from model import JointBert
from transformers import BertConfig, BertTokenizer, get_linear_schedule_with_warmup
from functions import IntentsAndSlots, collate_fn
from utils import Lang, load_data, save_model, get_splits, plot_model

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip):
    model.train()
    loss_array = []
    for sample in data:
        input_ids, attention_mask, token_type_ids = sample['input_ids'].to(device), sample['attention_mask'].to(device), sample['token_type_ids'].to(device)
        slot_labels = sample['slot_labels'].to(device)
        intent_labels = sample['intent_labels'].to(device)
        optimizer.zero_grad()
        slots, intents = model(input_ids, attention_mask, token_type_ids)
        loss_intent = criterion_intents(intents, intent_labels)
        loss_slot = criterion_slots(slots, slot_labels)
        loss = loss_intent + loss_slot
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("NaNs or Infs in loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"NaNs or Infs in gradients of {name}")
        optimizer.step()
        loss_array.append(loss.item())
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    special_tokens = [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            input_ids, attention_mask, token_type_ids = sample['input_ids'].to(device), sample['attention_mask'].to(device), sample['token_type_ids'].to(device)
            slot_labels = sample['slot_labels'].to(device)
            intent_labels = sample['intent_labels'].to(device)
            slots, intents = model(input_ids, attention_mask, token_type_ids)
            loss_intent = criterion_intents(intents, intent_labels)
            loss_slot = criterion_slots(slots, slot_labels)
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())

            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intent_labels'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            if -1 in output_slots:
                print("Error in slot labels")
                print(output_slots.shape)
                print(output_slots)

            # check this part
            for id_seq, seq in enumerate(output_slots):
                mask = ~np.isin(slot_labels[id_seq].cpu().numpy(), special_tokens)
                indices = list(np.where(mask)[0])
                utt_ids = [input_ids[id_seq][i].item() for i in indices]
                gt_ids = [slot_labels[id_seq][i].item() for i in indices]
                gt_slots = [lang.id2slot[x] for x in gt_ids]
                utterance = tokenizer.convert_ids_to_tokens(utt_ids)
                to_decode = [seq[i].item() for i in indices]

                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
                if len(ref_slots[id_seq]) != len(hyp_slots[id_seq]):
                    print("Error in slot labels")

    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

def main(train_loader, dev_loader, _, lang, model, optimizer, scheduler, criterion_slots, criterion_intents, clip=1, device='mps'):
    print("Eval before Training")
    results_test, intent_test, loss_array = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])
    
    top_f1 = 0
    try:
        n_epochs = 20
        patience = 4
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        loop = tqdm(range(0, n_epochs))
        for x in loop:
            loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip)
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(np.asarray(loss).mean())
                else:
                    scheduler.step()
            if x % 1 == 0:
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['total']['f']
            top_f1 = max(top_f1, f1)
            if f1 == top_f1:
                patience = 3
            else:
                patience -= 1

            if patience == 0:
                print("Early stopping")
                break
            loop.set_description(f"Epoch {x} - Train Loss: {losses_train[-1]:.4f} - Dev Loss: {losses_dev[-1]:.4f} - F1: {f1:.4f}")
            loop.set_postfix(slots=results_dev['total']['f'], intent=intent_res['accuracy'])
    except KeyboardInterrupt:
        print("Training interrupted")
        print("Eval after Training")
        results_test, intent_test, loss_array = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)

    plot_model(losses_train, losses_dev, sampled_epochs, dir='plots/')
    idx = get_index('bin/')
    save_model(model, optimizer, f'bin/model{idx}.pth')

    return results_test, intent_test, losses_train, losses_dev

if __name__ == '__main__':
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    config = BertConfig.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dev_data = load_data('dataset/ATIS/train.json')
    test_data = load_data('dataset/ATIS/test.json')
    train_raw, dev_raw, test_raw, y_train, y_dev, y_test = get_splits(train_dev_data, test_data)

    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, 
                                            # however this depends on the research purpose
    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=0)
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)


    train_dataset = IntentsAndSlots(train_raw, lang, tokenizer)
    dev_dataset = IntentsAndSlots(dev_raw, lang, tokenizer)
    test_dataset = IntentsAndSlots(test_raw, lang, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    lr = 2e-5 # learning rate
    clip = 1.0 # Clip the gradient
    dropout = 0.1 # Dropout rate
    model = JointBert(config, out_slot, out_int, dropout=dropout).to(device)
    model.bert.resize_token_embeddings(len(tokenizer))
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
    scheduler = None
    lrs = [1e-5, 2e-5, 3e-5, 4e-5]
    total_devs = []
    total_trains = []
    torch.save(model.state_dict(), 'tmp/initial_state.pth')
    with open('results.txt', 'w') as f:
        f.write("Results\n")
        for lr in lrs:
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            model.load_state_dict(torch.load('tmp/initial_state.pth'))
            print(f"Training with LR: {lr}")
            results_test, intent_test, losses_train, losses_dev = main(train_loader, dev_loader, test_loader, lang, model, optimizer, scheduler, criterion_slots, criterion_intents, clip, device)
            
            f.write(f"LR: {lr}\n")
            f.write(f"Slots: {results_test['total']['f']}\n")
            f.write(f"Intent: {intent_test['accuracy']}\n")
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
            total_devs.append(losses_dev)
            total_trains.append(losses_train)
    
    fig = plt.figure(figsize=(10,5))
    for id_lr, (lr, train, dev) in enumerate(zip(lrs, total_trains, total_devs)):
        plt.plot(train, label=f"Train - {lr}")
        plt.plot(dev, label=f"Dev - {lr}", linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses')
    plt.legend()
    idx = get_index('plots/comparisons')
    plt.savefig(f'plots/comparisons/plot_{idx}.png')
    with open(f"plots/pickle/comparison_{idx}.pkl", "wb") as f:
        pickle.dump((lrs, total_trains, total_devs), f)

    print("Done")