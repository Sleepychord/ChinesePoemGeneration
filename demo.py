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

start_token = u'B'
end_token = u'E'

def calc_word_freq(file_name):
    word_freq = {}

    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            # try :
            line_s = line.strip()
            title, content = line_s.split(u':')[-2:]
            content = content.replace(u' ', u'')
            if u'_' in content or u'(' in content or u'（' in content or u'《' in content or u'[' in content or \
                start_token in content or end_token in content:
                continue
            if len(content) < 5 or len(content) > 79:
                continue

            for sep in [u'，', u'。', u'？', u'！']:
                content = content.replace(sep, u'')

            if content[0] not in word_freq:
                word_freq[content[0]] = 1
            else:
                word_freq[content[0]] += 1

    return np.array(list(word_freq.keys())), np.array(list(word_freq.values()))

def prob_sample(weights, topn = 100):
    idx = np.argsort(weights)[::-1]
    t = np.cumsum(weights[idx[:topn]])
    s = np.sum(weights[idx[:topn]])
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    return idx[sample]

def evaluate(poems):
    scores = []
    for poem in poems:
        dic = {}
        for word in poem:
            dic[word] = 1
        scores.append(len(dic))
    return poems[np.argmax(scores)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dir", type = str, default = './model', help='file directory')
    parser.add_argument('-n', "--name", type=str, default='production.pth', help='file name')
    args = parser.parse_args()
    print(args)

    checkpoint = torch.load(os.path.join(args.dir, args.name))
    model, final, words, word2int, emb = checkpoint['model'], checkpoint['final'], checkpoint['words'], checkpoint['word2int'], checkpoint['emb']
    print('Finish Loading')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    final.to(device)
    start_words, start_freq = calc_word_freq('./data/poems.txt')
    while True:
        start = input()
        try:
            if len(start) == 0:
                start = start_words[prob_sample(start_freq)]
            poems = infer(model, final, words, word2int, emb, hidden_size = model.hidden_size, start=start, n = 20, num = 5 if random.random() < 0.5 else 7)
            print(evaluate(poems))
        except KeyError:
            print(u'此字在语料库中未出现过，请更换首字')
