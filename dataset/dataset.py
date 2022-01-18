# coding=utf8
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class CustomDataset(Dataset):
    """read all data into memory"""
    def __init__(self, dataset, column='SMILES'):
        """
        Args:
            dataset: smiles filename
            column: column name
        """
        super(CustomDataset, self).__init__()
        self.data = pd.read_csv(dataset)[column].tolist()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DataCollator:
    """collate_fn for SmilesDataset
       add_delimiter->translate->padding
    """
    def __init__(self):
        pass

    def __call__(self, data):
        """collate data to tensors
        Args:
            data: batch data from dataset
        Returns:
            padded_data: data padded
            lengths: seq length for packing
        """
        return data
