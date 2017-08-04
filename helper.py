import os
import pickle
import copy
import numpy as np
import json


CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }


# def load_data(path):
#     """
#     Load Dataset from File
#     """
#     input_file = os.path.join(path)
#     with open(input_file, 'r', encoding='utf-8') as f:
#         data = f.read()

#     return data

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


def preprocess_and_save_data(data_path, text_to_ids):
    """
    Preprocess Text Data.  Save to to file.
    """
    # Preprocess
    target_text, source_text = load_data(data_path)

    
    target_text = [article.lower() for article in target_text]
    source_text = [article.lower() for article in source_text]
    
    source_vocab_to_int, source_int_to_vocab = create_lookup_tables(source_text)
    target_vocab_to_int, target_int_to_vocab = create_lookup_tables(target_text)

    source_text, target_text = text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int)

    # Save Data
    pickle.dump((
        (source_text, target_text),
        (source_vocab_to_int, target_vocab_to_int),
        (source_int_to_vocab, target_int_to_vocab)), open('preprocess.p', 'wb'))


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
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


def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))


def batch_data(source, target, batch_size):
    """
    Batch source and target together
    """
    for batch_i in range(0, len(source)//batch_size):
        start_i = batch_i * batch_size
        source_batch = source[start_i:start_i + batch_size]
        target_batch = target[start_i:start_i + batch_size]
        yield np.array(pad_sentence_batch(source_batch)), np.array(pad_sentence_batch(target_batch))


def pad_sentence_batch(sentence_batch):
    """
    Pad sentence with <PAD> id
    """
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [CODES['<PAD>']] * (max_sentence - len(sentence))
            for sentence in sentence_batch]