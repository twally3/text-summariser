import json
import copy

CODES = { '<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }
WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '

def load_data(src):
    data = None
    heads, bodys = [], []

    with open(src) as data:
        data = json.load(data)

    for i in range(len(data)):
        heads.append(data[i]["head"])
        bodys.append(data[i]["body"])

    assert len(bodys) == len(heads)

    return heads, bodys

# def filter(line, whitelist):
#     return ''.join([ch for ch in line if ch in whitelist])

def preprocess_and_save_data(source_path, target_path, text_to_ids):
    """
    Preprocess Text Data.  Save to to file.
    """
    # Preprocess
    source_text = load_data(source_path)
    target_text = load_data(target_path)

    source_text = source_text.lower()
    target_text = target_text.lower()

    source_vocab_to_int, source_int_to_vocab = create_lookup_tables(source_text)
    target_vocab_to_int, target_int_to_vocab = create_lookup_tables(target_text)

    source_text, target_text = text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int)

    # Save Data
    pickle.dump((
        (source_text, target_text),
        (source_vocab_to_int, target_vocab_to_int),
        (source_int_to_vocab, target_int_to_vocab)), open('preprocess.p', 'wb'))


def load_preprocess():
    return pickle.load(open('preprocess.p', mode='rb'))

def create_lookup_tables(text):
    # text = [filter(article, WHITELIST) for article in text]
    vocab = set()

    for article in text:
        for word in article.split():
            vocab.add(word)

    vocab_to_int = copy.copy(CODES)

    for i, v in enumerate(vocab, len(CODES)):
        vocab_to_int[v] = i

    int_to_vocab = {i: v for v, i in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab