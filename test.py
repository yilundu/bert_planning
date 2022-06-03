from torch.utils.data import DataLoader
import torch
from data import Vocab, CLS, SEP, MASK, GraphDataset, GraphVocab
import numpy as np
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt
import seaborn as sns

# np.random.seed(15)
random.seed(1)

from bert import BERTLM


def refine_plan(traj, model, min_dist, dist_matrix, vocab):
    mask_token = vocab.token2idx(MASK)
    current_dists = []
    n = 500

    for i in range(500):
        replan_node = random.randint(1, traj.shape[0]-2)
        traj_candidate = torch.Tensor(traj)
        traj_candidate[replan_node] = mask_token

        traj_candidate = torch.Tensor(traj_candidate)[None, :].cuda().long()
        logit = model.work(traj_candidate)
        score = logit[0, replan_node, 5:]

        prev_node = int(vocab.idx2token(traj_candidate[0, replan_node - 1]))
        dist_score = torch.Tensor(dist_matrix[prev_node]).cuda()

        total_score = (score * 0.1 - dist_score * 3.0) * ((i+1) ** 0.5)
        prob = torch.softmax(total_score, dim=0)

        # print(i, score, dist_score)
        token_replace = torch.multinomial(prob, 1).item()

        token_replace = vocab.token2idx(str(token_replace))
        traj[replan_node] = token_replace

        current_dist = 0

        for i in range(traj.shape[0] - 1):
            start = traj[i]
            end = traj[i+1]

            start_node = int(vocab.idx2token(start))
            end_node = int(vocab.idx2token(end))

            dist = graph_dist[start_node, end_node]

            current_dist = current_dist + dist

        current_dists.append(current_dist)
        # Resample token based off BERT likelihoods
        print("best plan/ current plan: ", min_dist, current_dist)

    plt.scatter(np.arange(n), np.ones(n) * min_dist)
    plt.scatter(np.arange(n), np.array(current_dists))
    plt.savefig("plan_iteration.png")


def convert_graph_to_dist(graph):
    dist = np.copy(graph)
    dist[graph==0] = 100.0
    dist = dist * (1 - np.eye(dist.shape[0]))

    return dist


if __name__ == "__main__":
    ckpt = torch.load("ckpt/epoch_batch_179999")
    args = ckpt['args']

    vocab = GraphVocab(args.graph_size, specials=[MASK])
    graph_dataset = GraphDataset(vocab, args.plan_length, args.graph_size)

    truth, inp, msk = graph_dataset[1]

    model = BERTLM(0, vocab, args.embed_dim, args.ff_embed_dim, args.num_heads, args.dropout, args.layers, args.approx).cuda()
    model.load_state_dict(ckpt['model'])

    graph = graph_dataset.graph
    dist_matrix = convert_graph_to_dist(graph)

    graph_dist, graph_predecessors = shortest_path(csgraph=dist_matrix, unweighted=False, directed=True, return_predecessors=True)

    start_node = int(vocab.idx2token(truth[0]))
    end_node = int(vocab.idx2token(truth[-1]))

    min_dist = graph_dist[start_node, end_node]
    refine_plan(truth, model, min_dist, dist_matrix, vocab)
