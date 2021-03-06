#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file was originally developed for EvalNE by Alexandru Cristian Mara

import argparse
import numpy as np
import networkx as nx
import ast
import time
from utils import AROPE

# Although the method says it requires python 3.5, it should be run with python 2.7 for optimal results!

def parse_args():
    """ Parses AROPE arguments. """

    parser = argparse.ArgumentParser(description="Run AROPE.")

    parser.add_argument('--inputgraph', nargs='?',
                        default='BlogCatalog.csv',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?',
                        default='network.emb',
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
                        help='Embedding dimension. Default is 128. If use_tsne then tSNE is used to \
                            reduce the dimensionality to match this parameter. The original embedding will \
                            be of "dimension_before_tsne".')

    parser.add_argument('--delimiter', default=',',
                        help='The delimiter used to separate numbers in input file. Default is `,`')

    parser.add_argument('--order', type=int, default=3,
                        help='Order of the proximity. Default is 3.')

    parser.add_argument('--weights', default='[1,0.1,0.01]',
                        help='The weights for high-order proximity as list of len `order`. Default is `[1,0.1,0.01]`.')

    return parser.parse_args()


def main(args):
    """ Compute embeddings using AROPE. """

    # Load edgelist
    oneIndx = False
    E = np.loadtxt(args.inputgraph, delimiter=args.delimiter, dtype=int)
    if np.min(E) == 1:
        oneIndx = True
        E -= 1

    # Create a graph
    G = nx.Graph()

    # Make sure the graph is unweighted
    G.add_edges_from(E[:, :2])

    # Get adj matrix of the graph and symmetrize
    tr_A = nx.adjacency_matrix(G, weight=None)

    # Compute embeddings
    weights = ast.literal_eval(args.weights)
    U_list, V_list = AROPE(tr_A, args.dimension, [args.order], [weights])

    # AROPE sometimes returns more dimensions than specified. This depends on
    # the dataset. Here we make sure the embedding dimension is what we expect.
    U_list[0] = U_list[0][:, : args.dimension]
    V_list[0] = V_list[0][:, : args.dimension]

    start = time.time()
    # Read the train edges and compute simmilarity
    if args.tr_e is not None:
        train_edges = np.loadtxt(args.tr_e, delimiter=args.delimiter, dtype=int)
        if oneIndx:
            train_edges -= 1
        scores = list()
        for src, dst in train_edges:
            scores.append(U_list[0][src].dot(V_list[0][dst].T))
        np.savetxt(args.tr_pred, scores, delimiter=args.delimiter)

        # Read the test edges and run predictions
        if args.te_e is not None:
            test_edges = np.loadtxt(args.te_e, delimiter=args.delimiter, dtype=int)
            if oneIndx:
                test_edges -= 1
            scores = list()
            for src, dst in test_edges:
                scores.append(U_list[0][src].dot(V_list[0][dst].T)) 
            np.savetxt(args.te_pred, scores, delimiter=args.delimiter)

    # If no edge lists provided to predict links, then just store the simmilarity matrix
    else:
        np.savetxt(args.output, U_list[0], delimiter=args.delimiter)


if __name__ == "__main__":
    args = parse_args()
    main(args)

