

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
                slots.append((tmp[0], tmp[1] if tmp[1] == 'O' else 'T'))
            all_slots.append(slots)
            sentences.append(sentence)
    return sentences, all_slots