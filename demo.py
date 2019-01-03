# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function
from codecs import open as open

import torch
from preprocess import process_poems, start_token
import pdb
import tqdm
import numpy as np
import argparse
import sys
import os
from preprocess import pos2PE
from main import infer
import random
torch.random.manual_seed(0)
random.seed(0)
if __name__ == "__main__":
    checkpoint = torch.load('./model/production.pth')
    model, final, words, word2int, emb = checkpoint['model'], checkpoint['final'], checkpoint['words'], checkpoint['word2int'], checkpoint['emb']
    print('Finish Loading')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    final.to(device)
    while True:
        start = input()
        try:
            print(infer(model, final, words, word2int, emb, hidden_size = model.hidden_size, start=start))
        except KeyError:
            print(u'此字在语料库中未出现过，请更换首字')
