import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from functions import get_data, Slots
from transformers import BertConfig, BertTokenizer
from model import SlotBert
from functions import split_data, evaluate
from tqdm import tqdm
from utils import plot_results, save_model

def train_loop(train_loader, model, optimizer, criterion_slots, clip, device):
    model.train()
    train_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        slot_labels = batch['slot_labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()
        slot_logits = model(input_ids, attention_mask)
        loss = criterion_slots(slot_logits, slot_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()

    return train_loss/len(train_loader)

def eval_loop(dev_loader, model, criterion_slots, device):
    model.eval()
    valid_loss = 0
    ref_slot = []
    hyp_slot = []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch['input_ids'].to(device)
            slot_labels = batch['slot_labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            slot_logits = model(input_ids, attention_mask)
            loss = criterion_slots(slot_logits, slot_labels)
            valid_loss += loss.item()
            
            predictions = torch.argmax(slot_logits, dim=1)
            
            slot_labels = slot_labels.cpu()
            predictions = predictions.cpu()
            ref_slot.append(slot_labels.numpy())
            hyp_slot.append(predictions.numpy())

        precision, recall, f1_score = evaluate(ref_slot, hyp_slot)

    return precision, recall, f1_score, valid_loss/len(dev_loader)

def main(train_loader, dev_loader, test_loader, optimizer, scheduler, model, criterion_slots, clip, device, epochs=20):
    patience = 4
    best_valid_loss = float('inf')
    train_losses = []
    dev_losses = []
    metrics = []
    print("Before training: ")
    precision, recall, f1_score, _ = eval_loop(test_loader, model, criterion_slots, device)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1_score)

    loop = tqdm(range(1, epochs+1), desc="Epoch: ")
    metrics.append((precision, recall, f1_score))
    try:
        for _ in loop:
            _train_loss = train_loop(train_loader, model, optimizer, criterion_slots, clip, device)
            train_losses.append(_train_loss)

            precision, recall, f1_score, _valid_loss = eval_loop(dev_loader, model, criterion_slots, device)
            dev_losses.append(_valid_loss)
            metrics.append((precision, recall, f1_score))
            scheduler.step(dev_losses[-1])

            loop.set_postfix(train_loss=train_losses[-1], valid_loss=dev_losses[-1], precision=precision, recall=recall, f1_score=f1_score)
            if _valid_loss < best_valid_loss:
                best_valid_loss = _valid_loss
                patience = 4
                torch.save(model.state_dict(), 'model.pt')
            if _valid_loss > best_valid_loss:
                patience -= 1
            if patience == 0:
                print("Early stopping")
                break
    except KeyboardInterrupt:
        print("Training interrupted")

    # test results
    precision, recall, f1_score, _ = eval_loop(test_loader, model, criterion_slots, device)
    metrics.append((precision, recall, f1_score))
    plot_results(train_losses, dev_losses, metrics)
    save_model(model, optimizer, dir='SA/bin')
    
if __name__ == "__main__":
    train_path = "SA/dataset/laptop14_train.txt"
    test_path = "SA/dataset/laptop14_test.txt"

    train_dev_data = get_data(train_path)
    train_data, dev_data = split_data(train_dev_data, split_ratio=1)
    test_data = get_data(test_path)
    
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = SlotBert.from_pretrained('bert-base-uncased', config=config, num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = Slots(train_data, tokenizer)
    # dev_dataset = Slots(dev_data, tokenizer)
    test_dataset = Slots(test_data, tokenizer)

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print("Using device: ", device)
    
    train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    # dev_loader = data.DataLoader(dev_dataset, batch_size=128, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=1)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=0)

    main(train_loader, test_loader, test_loader, optimizer, scheduler, model.to(device), criterion_slots, clip=1, device=device)