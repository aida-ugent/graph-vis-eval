#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np

# libraries for deepwalk
import os
import random
from argparse import FileType, ArgumentDefaultsHelpFormatter

from deepwalk import graph
from deepwalk import walks as serialized_walks
from gensim.models import Word2Vec
from deepwalk.skipgram import Skipgram

from six.moves import range

import psutil
from multiprocessing import cpu_count

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass


def parse_args():
    """ Parses Deepwalk arguments. """

    parser = argparse.ArgumentParser(description="Deepwalk",
                                     formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--inputgraph',
                        help='Input graph path')

    parser.add_argument('--output',
                        help='Path where the embeddings will be stored.')

    parser.add_argument('--tr_e', nargs='?', default=None,
                        help='Path of the input train edges. Default None (in this case returns embeddings)')

    parser.add_argument('--tr_pred', nargs='?', default='tr_pred.csv',
                        help='Path where the train predictions will be stored.')

    parser.add_argument('--te_e', nargs='?', default=None,
                        help='Path of the input test edges. Default None.')

    parser.add_argument('--te_pred', nargs='?', default='te_pred.csv',
                        help='Path where the test predictions will be stored.')

    parser.add_argument('--dimension', type=int, default=2,
                        help='Embedding dimension. Default is 2. If use_tsne then tSNE is used to \
                            reduce the dimensionality to match this parameter. The original embedding will \
                            be of "dimension_before_tsne".')

    parser.add_argument('--delimiter', default=',',
                        help='The delimiter used to separate numbers in \
                            input file. Default is `,`')
    
    # Deepwalk specific hyperparameters
    parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
                        help='Size to start dumping walks to disk, instead of keeping them in memory.')

    parser.add_argument('--number-walks', default=80, type=int,
                        help='Number of random walks to start at each node')

    parser.add_argument('--walk-length', default=40, type=int,
                        help='Length of the random walk started at each node')

    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of skipgram model.')

    parser.add_argument('--workers', default=1, type=int,
                        help='Number of parallel processes.')

    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for random walk generator.')

    parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                        help='Use vertex degree to estimate the frequency of nodes in the \
                            random walks. This option is faster than calculating the vocabulary.')

    return parser.parse_args()


# load_edgelist from DeepWalk does not allow to specify delimiter and header
def load_edgelist(file_, undirected=True, delimiter=" ", header='#'):
  G = graph.Graph()
  with open(file_) as f:
    for l in f:
        if l[0] == header:
            continue
        x, y = l.strip().split(delimiter)[:2]
        x = int(x)
        y = int(y)
        G[x].append(y)
        if undirected:
            G[y].append(x)
  
  G.make_consistent()
  return G


def main(args):
    """ Compute embeddings using Deepwalk. """
    G = load_edgelist(args.inputgraph, undirected=True, delimiter=args.delimiter)
    num_walks = len(G.nodes()) * args.number_walks
    data_size = num_walks * args.walk_length

    print("Number of nodes: {}".format(len(G.nodes())))
    print("Number of walks: {}".format(num_walks))
    print("Data size (walks*length): {}".format(data_size))

    if data_size < args.max_memory_data_size:
        print("Walking...")
        walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,
                                            path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))
        print("Training...")
        model = Word2Vec(walks, size=args.dimension, window=args.window_size, min_count=0, sg=1, hs=1, workers=args.workers)
    else:
        print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, args.max_memory_data_size))
        print("Walking...")

        walks_filebase = args.output + ".walks"
        walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
                                            path_length=args.walk_length, alpha=0, rand=random.Random(args.seed),
                                            num_workers=args.workers)

        print("Counting vertex frequency...")
        if not args.vertex_freq_degree:
            vertex_counts = serialized_walks.count_textfiles(walk_files, args.workers)
        else:
            # use degree distribution for frequency in tree
            vertex_counts = G.degree(nodes=G.iterkeys())

        print("Training...")
        walks_corpus = serialized_walks.WalksCorpus(walk_files)
        model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                        size=args.dimension,
                        window=args.window_size, min_count=0, trim_rule=None, workers=args.workers)

    emb = model.wv.syn0
    indexlist = [int(idx) for idx in model.wv.index2word]
    # sort embedding according to node index
    emb = emb[np.argsort(indexlist)]
    #model.wv.save_word2vec_format(args.output)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Read the train edges and run predictions
    if args.tr_e is not None:
        adj_rec = np.dot(emb, emb.T)
        train_edges = np.loadtxt(args.tr_e, delimiter=args.delimiter, dtype=int)
        preds = []
        for e in train_edges:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
        np.savetxt(args.tr_pred, preds, delimiter=args.delimiter)

        # Read the test edges and run predictions
        if args.te_e is not None:
            test_edges = np.loadtxt(args.te_e, delimiter=args.delimiter, dtype=int)
            preds = []
            for e in test_edges:
                preds.append(sigmoid(adj_rec[e[0], e[1]]))
            np.savetxt(args.te_pred, preds, delimiter=args.delimiter)

    if args.output is not None:
        np.savetxt(args.output, emb, delimiter=args.delimiter)


if __name__ == "__main__":
    args = parse_args()
    main(args)
