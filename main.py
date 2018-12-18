import torch
from preprocess import process_poems, start_token
import pdb
import tqdm
import argparse

def sequence_collate(batch):
    transposed = zip(*batch)
    # ret = [torch.nn.utils.rnn.pack_sequence(sorted(samples, key = len, reverse = True)) for samples in transposed]
    ret = [torch.nn.utils.rnn.pack_sequence(samples) for samples in transposed]
    return ret

def infer(model, final, words, word2int, emb, hidden_size = 256, start = '春'):
    n = 1
    h = torch.zeros((1, n, hidden_size))
    x = torch.nn.functional.embedding(torch.full((n,), word2int[start], dtype = torch.long), emb).unsqueeze(0)
    ret = [[start] for i in range(n)]
    for i in range(19):
        if torch.cuda.is_available():
            x, h = x.cuda(), h.cuda()
        x, h = model(x, h)
        # h = torch.rand((1, n, hidden_size))
        w = torch.argmax(final(x.view(-1, hidden_size)), dim = 1).cpu()
        x = torch.nn.functional.embedding(w, emb).unsqueeze(0)
        for j in range(len(w)):
            ret[j].append(words[w[j]])
    ret_list = []
    for i in range(n):
        # print("".join(ret[i]))
        ret_list.append("".join(ret[i]))
    return ret_list

def main(epoch = 10, batch_size = 4, hidden_size = 256):
    dataset, words, word2int = process_poems('./data/poems.txt', './data/sgns.sikuquanshu.word')
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=sequence_collate)
    model = torch.nn.RNN(input_size = dataset.emb_dim, hidden_size = hidden_size)
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
                    "example": infer(model, final, words, word2int, dataset.emb)
                }
                data_iter.write(str(post_fix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=10, help="number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="number of batch_size")
    parser.add_argument("-hs", "--hidden_size", type=int, default=256, help="hidden size of RNN")
    args = parser.parse_args()
    print(args)

    main(epoch=args.epoch, batch_size=args.batch_size, hidden_size=args.hidden_size)
