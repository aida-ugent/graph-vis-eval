#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import time
import os
import sys

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import scipy.sparse as sp
import argparse
import numpy as np
import networkx as nx

from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')

def parse_args():
    """ Parses AROPE arguments. """
    parser = argparse.ArgumentParser(description="Run GAE.")

    parser.add_argument('--inputgraph', type=str,
                        help='Input graph path')

    parser.add_argument('--output', type=str,
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
                        help='Embedding dimension. Default is 2.')

    parser.add_argument('--delimiter', default=',',
                        help='The delimiter used to separate node ids in input file. Default is ","')

    parser.add_argument('--verbose', default=False, action='store_true',
                        help="Print training loss and accuracy during optimization.")
    return parser.parse_known_args()


def main(args):
    """ Compute embeddings using GCN. """

    # Load edgelist
    oneIndx = False
    E = np.loadtxt(args.inputgraph, delimiter=args.delimiter, dtype=int)
    if np.min(E) == 1:
        oneIndx = True
        E -= 1

    # Create a graph
    G = nx.Graph()
    G.add_edges_from(E[:, :2])
    adj = nx.adjacency_matrix(G, weight=None)

    # Define placeholders
    placeholders = {
        'features': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
        'dropout': tf.compat.v1.placeholder_with_default (0., shape=())
    }

    adj_norm = preprocess_graph(adj)
    num_nodes = adj.shape[0]
    
    adj_label = adj + sp.eye(num_nodes)
    adj_label = sparse_to_tuple(adj_label)
        
    features = sp.identity(num_nodes)
    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create model
    model = None
    if FLAGS.model == 'gcn_ae':
        model = GCNModelAE(placeholders,
                           num_features,
                           features_nonzero)
    elif FLAGS.model == 'gcn_vae':
        model = GCNModelVAE(placeholders,
                            num_features,
                            num_nodes,
                            features_nonzero)

    pos_weight = float(num_nodes * num_nodes - adj.sum()) / adj.sum()
    norm = num_nodes * num_nodes / float((num_nodes * num_nodes - adj.sum()) * 2)

    # Optimizer
    with tf.name_scope('optimizer'):
        if FLAGS.model == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                              labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                          validate_indices=False), [-1]),
                              pos_weight=pos_weight,
                              norm=norm)
        elif FLAGS.model == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                           validate_indices=False), [-1]),
                               model=model,
                               num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Train model
    for epoch in range(FLAGS.epochs):
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]
        if args.verbose:
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost),
                  "train_acc=", "{:.5f}".format(avg_accuracy),
                  "time=", "{:.5f}".format(time.time() - t))

    # get embedding
    feed_dict.update({placeholders['dropout']: 0})
    emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Read the train edges and run predictions
    if args.tr_e is not None:
        adj_rec = np.dot(emb, emb.T)
        train_edges = np.loadtxt(args.tr_e, delimiter=args.delimiter, dtype=int)
        if oneIndx:
            train_edges -= 1
        preds = []
        for e in train_edges:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
        np.savetxt(args.tr_pred, preds, delimiter=args.delimiter)

        # Read the test edges and run predictions
        if args.te_e is not None:
            test_edges = np.loadtxt(args.te_e, delimiter=args.delimiter, dtype=int)
            if oneIndx:
                test_edges -= 1
            preds = []
            for e in test_edges:
                preds.append(sigmoid(adj_rec[e[0], e[1]]))
            np.savetxt(args.te_pred, preds, delimiter=args.delimiter)

    if args.output is not None:
        np.savetxt(args.output, emb, delimiter=args.delimiter)


if __name__ == "__main__":    
    try:
        argv = FLAGS(sys.argv, known_only=True)
    except flags.Error as e:
        print('%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS))
        sys.exit(1)

    args, _ = parse_args()
    flags.DEFINE_integer('hidden2', args.dimension, 'Number of units in hidden layer 2.')
    main(args)
