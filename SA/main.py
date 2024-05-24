import os
from functions import get_data, Slots
from transformers import BertConfig, BertTokenizer
from model import SlotBert

if __name__ == "__main__":
    train_path = "SA/dataset/laptop14_train.txt"
    test_path = "SA/dataset/laptop14_test.txt"

    train_data = get_data(train_path)
    test_data = get_data(test_path)
    
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = SlotBert.from_pretrained('bert-base-uncased', config=config, num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = Slots(train_data, tokenizer)
    test_dataset = Slots(test_data, tokenizer)