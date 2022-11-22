import torch
import config
import pandas as pd
import torch.nn.functional as F


def get_dataset(csv_file, drop_low_samples=True):
    dataset = pd.read_csv(csv_file)
    cleaned_data = dataset.dropna()
    cleaned_data = cleaned_data[cleaned_data['IDENTITY'] != 'UNREADABLE']
    cleaned_data['IDENTITY'] = cleaned_data['IDENTITY'].str.lower()
    cleaned_data = cleaned_data.reset_index(drop=True)
    if drop_low_samples:
        indices = [idx for idx, label in enumerate(
            cleaned_data['IDENTITY'].values) if len(label) > config.MAX_LENGTH]
        cleaned_data = cleaned_data.drop(index=indices)
        cleaned_data = cleaned_data.reset_index(drop=True)
    return cleaned_data


def get_vocabulary():
    vocabulary = [' ', "'", '-', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                  'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    int2char = dict(enumerate(vocabulary))
    int2char = {k+1: v for k, v in int2char.items()}
    char2int = {v: k for k, v in int2char.items()}

    return int2char, char2int


def encode(string):
    _, char2int = get_vocabulary()
    token = torch.tensor([char2int[i] for i in string])
    pad_token = F.pad(token, pad=(0, config.MAX_LENGTH-len(token)),
                      mode='constant', value=0)
    return pad_token


def decode(token, vocabulary):
    int2char, _ = get_vocabulary()
    token = token[token != 0]
    string = [int2char[i.item()] for i in token]
    return "".join(string)
