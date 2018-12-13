import torch
from preprocess import process_poems, start_token
import pdb
def sequence_collate(batch):
    transposed = zip(*batch)
    # ret = [torch.nn.utils.rnn.pack_sequence(sorted(samples, key = len, reverse = True)) for samples in transposed]
    ret = [torch.nn.utils.rnn.pack_sequence(samples) for samples in transposed]
    return ret

def infer(model, final, words, word2int, emb, hidden_size = 256, start = '春'):
    n = 4
    h = torch.zeros((1, n, hidden_size))
    x = torch.nn.functional.embedding(torch.full((n,), word2int[start], dtype = torch.long), emb).unsqueeze(0)
    ret = [[start] for i in range(n)]
    for i in range(19):
        x, h = model(x, h)
        # h = torch.rand((1, n, hidden_size))
        w = torch.argmax(final(x.view(-1, hidden_size)), dim = 1)
        x = torch.nn.functional.embedding(w, emb).unsqueeze(0)
        for j in range(len(w)):
            ret[j].append(words[w[j]])
    for i in range(n):
        print("".join(ret[i]))

def main(batch_size = 4, hidden_size = 256):
    dataset, words, word2int = process_poems('./data/poems.txt', './data/sgns.sikuquanshu.word')
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=sequence_collate)
    model = torch.nn.RNN(input_size = dataset.emb_dim, hidden_size = hidden_size)
    final = torch.nn.Linear(hidden_size, dataset.voc_size, bias=False)
    opt = torch.optim.Adam(list(model.parameters()) + list(final.parameters()))
    for epoch in range(10):
        for data, label in loader:
            # pdb.set_trace()
            pred, _ = model(data)
            pred = final(pred.data)
            loss = torch.nn.functional.cross_entropy(pred, label.data)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(loss)
            infer(model, final, words, word2int, dataset.emb)    

if __name__ == "__main__":
    main()
