import os
from functions import get_data
from transformers import BertConfig
from model import SlotBert

if __name__ == "__main__":
    train_path = "SA/dataset/laptop14_train.txt"
    test_path = "SA/dataset/laptop14_test.txt"

    train_sentences, train_slots = get_data(train_path)
    test_sentences, test_slots = get_data(test_path)
    assert len(train_sentences) == 3045
    assert len(test_sentences) == 800


    config = BertConfig.from_pretrained('bert-base-uncased')
    model = SlotBert.from_pretrained('bert-base-uncased', config=config, num_labels=3)
