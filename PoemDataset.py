# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function
from codecs import open as open

import torch
from torch.utils.data import Dataset
import re


class PoemDataset(Dataset):
    def __init__(self, poems, emb):
        # self.poems = sorted(poems, key=len, reverse=True)
        self.poems = poems
        self.emb = emb
        self.voc_size, self.emb_dim = self.emb.size()
        self.emb_dim += self.poems[0].size()[1] - 1

    def __getitem__(self, index):
        poem = self.poems[index]
        y = poem[1:, 0].type(torch.long)
        x1 = poem[:-1, 0].type(torch.long)
        x1 = torch.nn.functional.embedding(x1, self.emb)
        x2 = poem[:-1, 1:]
        return torch.cat((x1, x2), 1), y

    def __len__(self):
        return len(self.poems)
