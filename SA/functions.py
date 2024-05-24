import torch.utils.data as data
import torch

class Lang():
    def __init__(self, slot2id, id2slot):
        self.slot2id = slot2id
        self.id2slot = id2slot

class Slots(data.Dataset):
    def __init__(self, data, tokenizer, max_seq_len=100):
        self.data = data
        self.tokenizer = tokenizer
        slot2id = {tokenizer.pad_token:tokenizer.pad_token_id,'T':1, 'O':2}
        id2slot = {tokenizer.pad_token_id:tokenizer.pad_token, 1:'T', 2:'O'}
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
                'attention_mask': torch.tensor(self.attention_mask)}

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