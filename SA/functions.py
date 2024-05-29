import torch.utils.data as data
import torch
from transformers import BertTokenizer
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def evaluate(ref_aspects, pred_aspects):
    ref_aspects = np.vstack(ref_aspects)
    pred_aspects = np.vstack(pred_aspects)

    assert len(ref_aspects) == len(pred_aspects)
    n_samples = len(ref_aspects)
    # number of true positive, gold standard, predicted opinion targets
    n_tp_aspects, n_gt_aspects, n_pred_aspects = 0, 0, 0
    for i in range(n_samples):
        gt_aspects = ref_aspects[i]
        p_aspects = pred_aspects[i]

        # Filter out PAD tokens (==0)
        indices = np.where(gt_aspects != 0)
        gt_aspects = gt_aspects[indices] - 1
        p_aspects = p_aspects[indices] - 1  

        # hit number
        n_hit = 0
        for pred, gt in zip(p_aspects, gt_aspects):
            if pred == 1 and gt == 1:
                n_hit += 1

        n_tp_aspects += n_hit
        n_gt_aspects += sum(gt_aspects) # TP + FN
        n_pred_aspects += sum([1 if x == 1 else 0 for x in p_aspects ]) # TP + FP
    # add 0.001 for smoothing
    # calculate precision, recall and f1 for ote task
    precision = float(n_tp_aspects) / float(n_pred_aspects + 1e-4)
    recall = float(n_tp_aspects) / float(n_gt_aspects + 1e-4)
    f1 = 2 * precision * recall / (precision + recall + 1e-4)
    scores = (precision, recall, f1)
    return scores

# def evaluate(ref_aspects, pred_aspects):
#     ref_aspects = np.vstack(ref_aspects).flatten()
#     pred_aspects = np.vstack(pred_aspects).flatten()

#     indices = ref_aspects != 0
#     ref_aspects = ref_aspects[indices]
#     pred_aspects = pred_aspects[indices]
#     assert len(ref_aspects) == len(pred_aspects)

#     tp = np.sum((ref_aspects == 2) & (pred_aspects == 2))
#     gt = np.sum(ref_aspects == 2)
#     pred = np.sum(pred_aspects == 2)

#     precision = tp / (pred + 1e-3)
#     recall = tp / (gt + 1e-3)
#     f1 = 2 * precision * recall / (precision + recall + 1e-3)

#     return precision, recall, f1
        
class Lang():
    def __init__(self, slot2id, id2slot):
        self.slot2id = slot2id
        self.id2slot = id2slot

class Slots(data.Dataset):
    def __init__(self, data, tokenizer, max_seq_len=100):
        self.data = data
        self.tokenizer = tokenizer
        slot2id = {tokenizer.pad_token:tokenizer.pad_token_id,'T':2, 'O':1}
        id2slot = {tokenizer.pad_token_id:tokenizer.pad_token, 2:'T', 1:'O'}
        lang = Lang(slot2id, id2slot)
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        unk_token = tokenizer.unk_token
        pad_token_id = tokenizer.pad_token_id  
        
        input_ids = []
        slot_labels = []
        input_ids = []
        slot_labels = []
        self.input_ids = []
        self.slot_labels = []
        self.attention_mask = []

        for sample in data:
            tokens = []
            slot_labels = []
            attention_mask = []
            sentence, slot = sample

            for word, slot in zip(sentence.split(" "), slot):
                _tokens = tokenizer.tokenize(word)
                if not _tokens:
                    _tokens = [unk_token]
                tokens.extend(_tokens)
                slot_labels.extend([lang.slot2id[slot]] + [pad_token_id]*(len(_tokens)-1))
            
            if len(tokens) > max_seq_len:
                print("Choose higher max_seq_len")
                tokens = tokens[:max_seq_len]
                slot_labels = slot_labels[:max_seq_len]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)

            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            if (len(slot_labels) + padding_length != max_seq_len):
                print("Smth wrong")
            slot_labels = slot_labels + ([pad_token_id] * padding_length)
            if len(attention_mask) < max_seq_len:
                attention_mask = attention_mask + ([0] * padding_length)
            

            self.input_ids.append(input_ids)
            self.slot_labels.append(slot_labels)
            self.attention_mask.append(attention_mask)
            
        assert len(self.input_ids[-1]) == max_seq_len
        assert len(self.slot_labels[-1]) == max_seq_len
        assert len(self.attention_mask[-1]) == max_seq_len
        assert len(self.input_ids) == len(self.slot_labels) == len(self.attention_mask)
        
        print("Loaded dataset")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.input_ids[idx]),
                'slot_labels': torch.tensor(self.slot_labels[idx]),
                'attention_mask': torch.tensor(self.attention_mask[idx])}

def get_data(path):
    sentences = []
    all_slots = []
    with open(path, 'r') as f:
        rows = f.readlines()
        for row in rows:
            sentence, _slots= row.split("####")
            sentence = sentence.strip()
            _slots = _slots.split(" ")
            slots = []
            for slot in _slots:
                tmp = slot.rsplit('=', 1)
                slots.append(tmp[1] if tmp[1] == 'O' else 'T')
            all_slots.append(slots)
            sentences.append(sentence)

    data = list(zip(sentences, all_slots))
    return data

def split_data(data, split_ratio=0.8):
    n = len(data)
    train_data = data[:int(n*split_ratio)]
    test_data = data[int(n*split_ratio):]
    return train_data, test_data