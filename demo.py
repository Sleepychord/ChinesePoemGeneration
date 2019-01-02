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

if __name__ == "__main__":
    dataset, words, word2int = process_poems('./data/poems.txt', './data/sgns.sikuquanshu.word')
    checkpoint = torch.load('./model/production.pth')
    model, final = checkpoint['model'], checkpoint['final']
    while True:
        start = input()
        print(infer(model, final, wordsk, word2int, dataset.emb, start=start))