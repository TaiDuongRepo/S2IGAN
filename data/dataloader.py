import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


def sen_collate_fn(batch):
    imgs = torch.stack([i[0] for i in batch])
    specs = pad_sequence([i[1] for i in batch], batch_first=True)  # (-1, len, n_mels)
    len_specs = torch.LongTensor([i[2] for i in batch])
    labels = torch.LongTensor([i[3] for i in batch])

    specs = specs.permute(0, 2, 1)  # (-1, n_mels, len)

    return imgs, specs, len_specs, labels
