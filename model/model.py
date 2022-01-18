# coding=utf8
import time

import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.linear = nn.Linear(config.feat_dim, config.out_dim)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def forward(self, x):
        """
        Args:
            x: input
        Return:
            logits: prediction
        """
        logits = self.linear(x)
        logits = self.dropout(logits)

        return logits

    def sample(self):
        pass