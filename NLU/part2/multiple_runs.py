from main2 import main
import matplotlib.pyplot as plt
import torch
from transformers import BertConfig, BertTokenizer
from functions import IntentsAndSlots, collate_fn
from utils import Lang, load_data, get_splits, get_index
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from model import JointBert
from transformers import get_linear_schedule_with_warmup


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device[{}]".format(device))

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

    ## Make changes from here onwards

    lr = 2e-5 # learning rate
    clip = 1.0 # Clip the gradient
    dropout = 0.1 # Dropout rate
    model = JointBert(config, out_slot, out_int, dropout=dropout).to(device)
    model.bert.resize_token_embeddings(len(tokenizer))
    
    criterion_slots = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

    lrs = [1e-5, 2e-5, 3e-5, 4e-5]
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    schedulers = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True),
                  get_linear_schedule_with_warmup(optimizer, num_warmup_steps=3, num_training_steps=20),
                  optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1),
                  optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=1)] # No scheduler
    scheduler_names = ['ReduceLROnPlateau', 'LinearWarmup', 'StepLR', 'NoScheduler']
    # Plot all results together
    
    plt.figure(figsize=(10, 6))
    for lr in lrs:
        for g in optimizer.param_groups:
            g['lr'] = lr
        for scheduler, sched_name in zip(schedulers, scheduler_names):
            try:
                results_test, intent_test, losses_train, losses_dev = main(train_loader,
            dev_loader, test_loader, lang, model, optimizer, scheduler, criterion_slots,
            criterion_intents, clip, device)
            except KeyboardInterrupt:
                print("Training interrupted")
            except Exception as e:
                print(f"Error: {e}")
            
            if losses_train is not None and losses_dev is not None:
                if len(losses_train) == len(losses_dev):
                    plt.plot(losses_train, label=f"DEV:lr={lr}, {sched_name}")
                    plt.plot(losses_dev, label=f"TRAIN:lr={lr}, {sched_name}")
    
    