# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function
from io import open as open

import collections
import os
import sys
import numpy as np
from collections import defaultdict
import re
from PoemDataset import PoemDataset
import torch
import random

torch.random.manual_seed(0)
random.seed(0)
start_token = u'B'
end_token = u'E'
invalid_chr = [u'_', u'(', u'（', u'《', u'[', start_token, end_token]

dim_PE = 100
PE_const = 1000
PE_tmp_divider = [float(np.power(PE_const, i / float(dim_PE))) for i in range(dim_PE)]


def pos2PE(pos):
    PE_tmp = pos * np.ones(dim_PE) / PE_tmp_divider
    PE_tmp[0::2] = np.sin(PE_tmp[0::2])
    PE_tmp[1::2] = np.cos(PE_tmp[1::2])
    return PE_tmp


def process_poems(file_name, embedding_file_name):
    # poems -> list of numbers
    poems = []
    poems_PE = []

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
            # content = start_token + re.sub('[，。]', '', content) + end_token
            # if len(re.split("，|。|？|！", content)[0]) != 5:
            #     continue
            pos = 0
            pos_PE = []
            sep_list = [u'，', u'。', u'？', u'！']
            for i, word in enumerate(content):
                if word in sep_list:
                    pos = 0
                else:
                    pos += 1
                    pos_PE.append(pos2PE(pos))
            for sep in [u'，', u'。', u'？', u'！']:
                content = content.replace(sep, u'')

            poems_PE.append(np.array(pos_PE))
            # poems_pos.append(pos_poem)
            poems.append(content)
    # except ValueError as e:
    # 	pass

    words = list(set([word for poem in poems for word in poem]))

    print('Reading embedding...')
    with open(embedding_file_name, 'r') as f:
        n, m = map(int, f.readline().split())
        emb_dict = defaultdict(lambda: np.random.normal(0, 1, size=(m,)))
        for line in f.readlines():
            emb = line.split()
            key = emb.pop(0)
            emb_dict[key] = np.array(emb, dtype=float)

    voc_size = len(words)
    emb_dim = m

    emb = torch.zeros((voc_size, emb_dim), requires_grad=False)
    word2int = {}
    for i, word in enumerate(words):
        emb[i] = torch.tensor(emb_dict[word])
        word2int[word] = i

    poems_int = []
    for i in range(len(poems)):
        poem_int = torch.tensor([word2int[w] for w in poems[i]], dtype=torch.float).unsqueeze(1)
        # poem_pos = torch.tensor(poems_pos[i], dtype=torch.long)
        poem_PE = torch.tensor(poems_PE[i], dtype=torch.float)
        poems_int.append(torch.cat((poem_int, poem_PE), 1))

    return PoemDataset(poems_int, emb), words, word2int


if __name__ == "__main__":
    dataset, emb, words = process_poems('./data/poems.txt', './data/sgns.sikuquanshu.word')
    import pdb;

    pdb.set_trace()
