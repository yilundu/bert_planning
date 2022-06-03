import random
import torch
import numpy as np
import re
from google_bert import create_instances_from_document
from torch.utils.data import Dataset
import os.path as osp

def sample_graph(graph, plan_length, weighted=False):
    n, n = graph.shape
    node = np.random.randint(0, n)

    if not weighted:
        graph = graph > 1e-2
        graph = np.clip(graph + np.eye(n), 0, 1)
        graph = graph / (graph.sum(axis=-1)[:, None])

    nodes = [node]
    transition = graph[node]

    while len(nodes) < plan_length:
        node = np.random.choice(np.arange(n), p=transition)
        nodes.append(node)
        transition = graph[node]

    return nodes

class GraphDataset(Dataset):

    def __init__(self, vocab, plan_length=10, graph_size=10):
        self.vocab = vocab
        self.plan_length = plan_length
        self.graph_size = graph_size
        self.n_context = 10
        # filepath = "graph_{}.npy".format(self.graph_size)

        # if not osp.exists(filepath):
        #     graph = np.random.uniform(0, 1, (self.graph_size, self.graph_size))
        #     graph_edge = graph > 0.7
        #     graph = graph_edge * graph
        #     # graph = graph + graph.transpose()

        #     np.save(filepath, graph)

        # graph = np.load(filepath)
        # self.graph = graph

    def __len__(self):
        return int(1e6)

    def gen_graph(self):
        graph = np.random.uniform(0, 1, (self.graph_size, self.graph_size))
        graph_edge = graph > 0.8
        graph = graph_edge * graph
        return graph

    def encode(self, node):
        node_list = []
        for elem in node:
            if elem == self.plan_length:
                encode = self.vocab.token2idx(MASK)
            else:
                elem = str(elem)
                encode = self.vocab.token2idx(elem)

            node_list.append(encode)

        return np.array(node_list)

    def __getitem__(self, idx):
        graph = self.gen_graph()
        nodes = []

        for i in range(self.n_context):
            node = sample_graph(graph, self.plan_length)
            nodes.append(node)

        node_context = np.concatenate(nodes, axis=0)

        node = sample_graph(graph, self.plan_length)
        node = np.array(node)
        mask = np.random.uniform(0, 1, node.shape) > 0.5

        node_mask = np.copy(node)
        node_mask[mask] = self.plan_length

        node = np.concatenate([node_context, node], axis=0)
        node_mask = np.concatenate([node_context, node_mask], axis=0)
        mask = np.concatenate([np.zeros(node_context.shape[0]), mask]).astype(np.bool)

        node = self.encode(node)
        node_mask = self.encode(node_mask)

        return node, node_mask, mask


PAD, UNK, CLS, SEP, MASK, NUM, NOT_CHINESE = '<-PAD->', '<-UNK->', '<-CLS->', '<-SEP->', '<-MASK->', '<-NUM->', '<-NOT_CHINESE->'
BUFSIZE = 40960000

def ListsToTensor(xs, vocab=None):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        if vocab is not None:
            y = vocab.token2idx(x) + [vocab.padding_idx]*(max_len -len(x))
        else:
            y = x + [0]*(max_len -len(x))
        ys.append(y)
    data = torch.LongTensor(ys).t_().contiguous()
    return data

def random_mask(tokens, masked_lm_prob, max_predictions_per_seq, vocab):
    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    masked_tokens, mask = [], []
    cand = []
    for i, token in enumerate(tokens):
        if token == CLS or token == SEP:
            continue
        cand.append(i)
    random.shuffle(cand)
    cand = set(cand[:num_to_predict])

    masked_tokens, mask = [], []
    for i, token in enumerate(tokens):
        if i in cand:
            if random.random() < 0.8:
                masked_tokens.append(MASK)
            else:
                if random.random() < 0.5:
                    masked_tokens.append(token)
                else:
                    masked_tokens.append(vocab.random_token())
            mask.append(1)
        else:
            masked_tokens.append(token)
            mask.append(0)
    return masked_tokens, mask

def _back_to_text_for_check(x, vocab):
    w = x.t().tolist()
    for sent in vocab.idx2token(w):
        print (' '.join(sent))
    
def batchify(data, vocab):
    truth, inp, seg, msk = [], [], [], []
    nxt_snt_flag = []
    for a, b, r in data:
        x = [CLS]+a+[SEP]+b+[SEP]
        truth.append(x)
        seg.append([0]*(len(a)+2) + [1]*(len(b)+1))
        masked_x, mask = random_mask(x, 0.15, 20, vocab)
        inp.append(masked_x)
        msk.append(mask)
        if r:
            nxt_snt_flag.append(0)
        else:
            nxt_snt_flag.append(1)

    truth = ListsToTensor(truth, vocab)
    inp = ListsToTensor(inp, vocab)
    seg = ListsToTensor(seg)
    msk = ListsToTensor(msk).to(torch.uint8)
    nxt_snt_flag = torch.ByteTensor(nxt_snt_flag)
    return truth, inp, seg, msk, nxt_snt_flag

class DataLoader(object):
    def __init__(self, vocab, filename, batch_size, max_len):
        self.batch_size = batch_size
        self.vocab = vocab
        self.max_len = max_len
        self.filename = filename
        self.stream = open(self.filename, encoding='utf8')
        self.epoch_id = 0

    def __iter__(self):
        
        lines = self.stream.readlines(BUFSIZE)

        if not lines:
            self.epoch_id += 1
            self.stream.close()
            self.stream = open(self.filename, encoding='utf8')
            lines = self.stream.readlines(BUFSIZE)

        docs = [[]]
        for line in lines:
            tokens = line.strip().split()
            if tokens:
                docs[-1].append(tokens)
            else:
                docs.append([])
        docs = [x for x in docs if x]
        random.shuffle(docs)

        data = []
        for idx, doc in enumerate(docs):
            data.extend(create_instances_from_document(docs, idx, self.max_len))

        idx = 0
        while idx < len(data):
            yield batchify(data[idx:idx+self.batch_size], self.vocab)
            idx += self.batch_size

class GraphDataLoader(object):
    def __init__(self, vocab, filename, batch_size, max_len):
        self.batch_size = batch_size
        self.vocab = vocab
        self.max_len = max_len
        self.filename = filename
        self.stream = open(self.filename, encoding='utf8')
        self.epoch_id = 0

    def __iter__(self):
        
        lines = self.stream.readlines(BUFSIZE)

        if not lines:
            self.epoch_id += 1
            self.stream.close()
            self.stream = open(self.filename, encoding='utf8')
            lines = self.stream.readlines(BUFSIZE)

        docs = [[]]
        for line in lines:
            tokens = line.strip().split()
            if tokens:
                docs[-1].append(tokens)
            else:
                docs.append([])
        docs = [x for x in docs if x]
        random.shuffle(docs)

        data = []
        for idx, doc in enumerate(docs):
            data.extend(create_instances_from_document(docs, idx, self.max_len))

        idx = 0
        while idx < len(data):
            yield batchify(data[idx:idx+self.batch_size], self.vocab)
            idx += self.batch_size

class GraphVocab(object):
    def __init__(self, graphsize, specials = None):
        self.num_re = re.compile(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$")
        idx2token = [PAD, UNK, NUM, NOT_CHINESE] + ( specials if specials is not None else [])

        for i in range(graphsize):
            idx2token.append(str(i))

        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]
        self._num_idx = self._token2idx[NUM]

    @property
    def size(self):
        return len(self._idx2token)
    
    @property
    def unk_idx(self):
        return self._unk_idx
    
    @property
    def padding_idx(self):
        return self._padding_idx
    
    @property
    def num_idx(self):
        return self._num_idx

    @property
    def no_chinese_idx(self):
        return self._no_chinese_idx

    def random_token(self):
        return self.idx2token(1 + np.random.randint(self.size-1))

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        if x in self._token2idx:
            return self._token2idx[x]
        if self.num_re.match(x) is not None:
            return self.num_idx
        if _has_non_chinese_char(x):
            return self._no_chinese_idx
        return self.unk_idx


class Vocab(object):
    def __init__(self, filename, min_occur_cnt, specials = None):
        self.num_re = re.compile(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$")
        idx2token = [PAD, UNK, NUM, NOT_CHINESE] + ( specials if specials is not None else [])
        for line in open(filename, encoding='utf8').readlines():
            try: 
                token, cnt = line.strip().split()
            except:
                continue
            if self.num_re.match(token) is not None:
                continue
            if _has_non_chinese_char(token):
                if int(cnt) >= 2*min_occur_cnt:
                    idx2token.append(token)
            else:
                if int(cnt) >= min_occur_cnt:
                    idx2token.append(token)

        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]
        self._num_idx = self._token2idx[NUM]
        self._no_chinese_idx = self._token2idx[NOT_CHINESE]

    @property
    def size(self):
        return len(self._idx2token)
    
    @property
    def unk_idx(self):
        return self._unk_idx
    
    @property
    def padding_idx(self):
        return self._padding_idx
    
    @property
    def num_idx(self):
        return self._num_idx

    @property
    def no_chinese_idx(self):
        return self._no_chinese_idx

    def random_token(self):
        return self.idx2token(1 + np.random.randint(self.size-1))

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        if x in self._token2idx:
            return self._token2idx[x]
        if self.num_re.match(x) is not None:
            return self.num_idx
        if _has_non_chinese_char(x):
            return self._no_chinese_idx
        return self.unk_idx

def _has_non_chinese_char(s):
    for x in s:
        cp = ord(x)
        if not ((cp >= 0x4E00 and cp <= 0x9FFF) or
            (cp >= 0x3400 and cp <= 0x4DBF) or
            (cp >= 0x20000 and cp <= 0x2A6DF) or
            (cp >= 0x2A700 and cp <= 0x2B73F) or
            (cp >= 0x2B740 and cp <= 0x2B81F) or
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or
            (cp >= 0x2F800 and cp <= 0x2FA1F)):
            return True
    return False


if __name__ == "__main__":
    graph_vocab = GraphVocab(10, [MASK])
    dataset = GraphDataset(graph_vocab, 10, 10)
    node, node_mask, mask = dataset[0]
    import pdb
    pdb.set_trace()
    print(node)
    print(node_mask)
    print(mask)
