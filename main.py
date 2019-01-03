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
torch.manual_seed(0)

def sequence_collate(batch):
    transposed = zip(*batch)
    ret = [torch.nn.utils.rnn.pack_sequence(sorted(samples, key = len, reverse = True)) for samples in transposed]
    # ret = [torch.nn.utils.rnn.pack_sequence(samples) for samples in transposed]
    return ret


def prob_sample(w_list, topn = 10):
    samples = []
    for weights in w_list:
        idx = np.argsort(weights)[::-1]
        t = np.cumsum(weights[idx[:topn]])
        s = np.sum(weights[idx[:topn]])
        sample = int(np.searchsorted(t, np.random.rand(1) * s))
        samples.append(idx[sample])
    return np.array(samples)


def infer(model, final, words, word2int, emb, hidden_size=256, start=u'春', n=1):
    dim_PE = 100
    PE_const = 1000
    device = torch.device('cpu') if isinstance(final.weight, torch.FloatTensor) else final.weight.get_device()
    h = torch.zeros((1, n, hidden_size))
    x = torch.nn.functional.embedding(torch.full((n,), word2int[start[0]], dtype=torch.long), emb).unsqueeze(0)
    ret = [[start[0]] for i in range(n)]
    for i in range(19):
        # add PE dims
        pe = torch.tensor(pos2PE((i % 5) + 1), dtype=torch.float).repeat(1, n, 1)
        
        x, h, pe = x.to(device), h.to(device), pe.to(device)
        x = torch.cat((x, pe), dim=2)
        x, h = model(x, h)
        if i % 5 == 4 and i // 5 + 1 < len(start):
            w = np.array([word2int[start[i // 5 + 1]] for _ in range(n)])
        else:
            w = prob_sample(torch.nn.functional.softmax(final(x.view(-1, hidden_size)), dim=-1).data.cpu().numpy())
        x = torch.nn.functional.embedding(torch.from_numpy(w), emb).unsqueeze(0)
        for j in range(len(w)):
            ret[j].append(words[w[j]])
            if i % 5 == 3:
                if sys.version_info.major == 2:
                    ret[j].append(u"，" if i < 18 else u"。")
                else:
                    ret[j].append("，" if i < 18 else "。")
    ret_list = []
    for i in range(n):
        if sys.version_info.major == 2:
            ret_list.append(u"".join(ret[i]))
        else:
            ret_list.append("".join(ret[i]))
    return ret_list


def main(epoch=10, batch_size=4, hidden_size=256, save_dir='./model'):
    dataset, words, word2int = process_poems('./data/poems.txt', './data/sgns.sikuquanshu.word')
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=sequence_collate)
    model = torch.nn.GRU(input_size=dataset.emb_dim, hidden_size=hidden_size)
    final = torch.nn.Linear(hidden_size, dataset.voc_size, bias=False)
    opt = torch.optim.Adam(list(model.parameters()) + list(final.parameters()))
    if torch.cuda.is_available():
        model, final = model.cuda(), final.cuda()

    for epoch in range(epoch):
        data_iter = tqdm.tqdm(enumerate(loader),
                              desc="EP_%d" % (epoch),
                              total=len(loader),
                              bar_format="{l_bar}{r_bar}")
        for i, data in data_iter:
            data, label = data
            # pdb.set_trace()
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()
            pred, _ = model(data)
            pred = final(pred.data)
            loss = torch.nn.functional.cross_entropy(pred, label.data)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # print(loss)

            if i % 100 == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "loss": loss.item(),
                    "example": infer(model, final, words, word2int, dataset.emb, hidden_size = hidden_size)
                }
                if sys.version_info.major == 2:
                    data_iter.write(unicode(post_fix))
                else:
                    data_iter.write(str(post_fix))
        # break

        tmp_infer_rst = infer(model, final, words, word2int, dataset.emb, hidden_size = hidden_size, n=5)
        if sys.version_info.major == 2:
            tmp_infer_rst = u"\n".join(tmp_infer_rst).encode('utf-8')
        else:
            tmp_infer_rst = "\n".join(tmp_infer_rst)
        print(tmp_infer_rst)
    
    print('Saving...')
    torch.save({
        'model': model.cpu(),
        'final': final.cpu(),
        'words': words,
        'word2int': word2int,
        'emb': dataset.emb
    }, os.path.join(save_dir, 'current.pth'))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=10, help="number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="number of batch_size")
    parser.add_argument("-hs", "--hidden_size", type=int, default=256, help="hidden size of RNN")
    parser.add_argument('-d', "--save_dir", type = str, default = './model', help='directory to save files in')
    args = parser.parse_args()
    print(args)

    main(epoch=args.epoch, batch_size=args.batch_size, hidden_size=args.hidden_size, save_dir = args.save_dir)
