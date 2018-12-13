import collections
import os
import sys
import numpy as np
from collections import defaultdict
import re
from PoemDataset import PoemDataset
import torch
start_token = 'B'
end_token = 'E'


def process_poems(file_name, embedding_file_name):
    # poems -> list of numbers
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                # content = start_token + re.sub('[，。]', '', content) + end_token
                content = re.sub('[，。]', '', content)
                poems.append(content)
            except ValueError as e:
                pass

    words = list(set([word for poem in poems for word in poem]))
    
    print('Reading embedding...')
    with open(embedding_file_name, 'r') as f:
        n, m = map(int, f.readline().split())
        emb_dict = defaultdict(lambda : np.random.normal(0, 1, size = (m,)))
        for line in f.readlines():
            emb = line.split()
            key = emb.pop(0)
            emb_dict[key] = np.array(emb, dtype = float)

    voc_size = len(words)
    emb_dim = m

    emb = torch.zeros((voc_size, emb_dim), requires_grad = False)
    word2int = {}
    for i, word in enumerate(words):
        emb[i] = torch.tensor(emb_dict[word])
        word2int[word] = i
    
    poems_int = [torch.tensor([word2int[w] for w in poem], dtype = torch.long) for poem in poems]

    return PoemDataset(poems_int, emb), words, word2int

if __name__ == "__main__":
    dataset, emb, words, _ = process_poems('./data/poems.txt', './data/sgns.sikuquanshu.word')
    import pdb; pdb.set_trace()