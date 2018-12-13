import torch
from torch.utils.data import Dataset
import re
class PoemDataset(Dataset):
    def __init__(self, poems, emb):
        self.poems = sorted(poems, key = len, reverse = True)
        self.emb = emb
        self.voc_size, self.emb_dim = self.emb.size()
    def __getitem__(self, index):
        # return torch.nn.functional.embedding(self.poems[index][:-1], self.emb), self.poems[index][1:]
        return torch.nn.functional.embedding(self.poems[index][:-1], self.emb), self.poems[index][1:]
    def __len__(self):
        return len(self.poems)