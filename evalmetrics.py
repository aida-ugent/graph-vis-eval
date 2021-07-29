#!/usr/bin/env python
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import os
import re
import subprocess
from scipy.spatial.distance import pdist
from networkx.drawing.nx_pydot import write_dot
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import BallTree


def shortest_path_distances(G):
    return shortest_path(csgraph=nx.adjacency_matrix(G, nodelist=np.arange(G.number_of_nodes())),
                         directed=False, unweighted=True)[np.triu_indices(G.number_of_nodes(), k=1)]


def stress(G, emb, data_dict=None, **kwargs):
    """
    Compute the normalized stress metric. This involves computing shortest 
    path distances and pairwise embedding distances between ALL nodes.
    """
    print("Computing pairwise Euclidean distances...", end="")
    emb_dist = pdist(emb, metric="euclidean")
    print("Done.")

    if "graph_dist" in data_dict:
        graph_dist = data_dict["graph_dist"]
    elif "graph_dist_file" in data_dict and os.path.isfile(data_dict["graph_dist_file"]):
        print("Reading shortest path distances from file...", end="")
        graph_dist = np.loadtxt(
            data_dict["graph_dist_file"], delimiter=",", dtype=int)[:, 2]
        data_dict["graph_dist"] = graph_dist
        print("Done.")
    else:
        print("Computing shortest paths...", end="")
        graph_dist = shortest_path_distances(G)
        print("Done.")
        indices = np.triu_indices(G.number_of_nodes(), k=1)
        filename = os.path.join("data", data_dict["name"] + "_graph_distances.txt")
        np.savetxt(filename,
                   np.hstack((indices[0][:, np.newaxis], indices[1]
                             [:, np.newaxis], graph_dist[:, np.newaxis])),
                   fmt="%d",
                   delimiter=",",
                   header="node1,node2,spd")
        print(f"Saved graph shortest path distances to {filename}. " +
              f"Read them using 'graph_dist_file={filename}' in the {data_dict['name']} config for faster stress computation.")
        data_dict["graph_dist"] = graph_dist

    def stress_func(alpha):
        return np.sum((alpha * np.divide(emb_dist, graph_dist) - 1) ** 2)

    alpha = np.sum(np.divide(emb_dist, graph_dist)) / \
        np.sum(np.divide(emb_dist, graph_dist)**2)
    stress = stress_func(alpha) / len(graph_dist)
    return stress


def neighborhood_preservation(G, emb, k=2, **kwargs):
    """
    Compute the neighborhood preservation as defined in 
    'DRGraph: An Efficient Graph Layout Algorithm for Large-scale 
    Graphs by Dimensionality Reduction'. It is defined as the Jaccard 
    similarity between k-order neighborhoods in the the graph and 
    the embedding space.

    Parameters
    ----------
    G : nx.graph
        The original graph to use for the evaluation.
    emb : ndarray
        Low-dimensional embedding of all nodes in G.
    k : int, default 2
        Order to define the maximum hop-distance in G to obtain 
        the neighbors of each node. The node itself is not included 
        in the set of neighbors.

    Return
    -------
    neighborhood preservation score in [0,1]
    """
    num_nodes = G.number_of_nodes()
    similarity = 0.0
    tree = BallTree(emb, leaf_size=40)

    for i in range(num_nodes):
        G_neigh = nx.single_source_shortest_path_length(G, i, k)
        G_neigh.pop(i)
        G_neigh = G_neigh.keys()
        emb_neigh = tree.query(emb[i].reshape(
            1, -1), k=len(G_neigh)+1, return_distance=False)[0][1:]
        similarity += jaccard(G_neigh, emb_neigh)

    return similarity / num_nodes


def jaccard(a, b):
    intersection = len(list(set(a).intersection(b)))
    union = (len(a) + len(b)) - intersection
    return float(intersection) / union


def glam_scores(G, emb, glam_path=None, metrics="all", **kwargs):
    """
    Computes the scores using the glam implementation:

    Parameters
    ----------
    G : nx.graph
        The original graph to use for the evaluation.
    emb : ndarray
        Low-dimensional embedding of all nodes in G.
    glam_path: str
        Path to glam executable.

    Return
    -------
    dictionary with keys for each metric and values are the computed scores
    """
    # store embedding coordinates in networkx object
    pos_x = dict(zip(np.arange(0, len(G)), emb[:, 0]))
    pos_y = dict(zip(np.arange(0, len(G)), emb[:, 1]))

    # add embedding coordinates to copy of the graph
    tmp_G = G.copy()
    nx.set_node_attributes(tmp_G, pos_x, "x")
    nx.set_node_attributes(tmp_G, pos_y, "y")
    tmp_G_file = "tmp_G.dot"
    write_dot(tmp_G, tmp_G_file)

    if metrics == "all":
        metrics = ["crosslessness", "edge_length_cv",
                   "min_angle", "shape_gabriel"]

    cmd = [glam_path, tmp_G_file, "-m"] + metrics
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p1.wait()
    stdout, _ = p1.communicate()
    stdout = stdout.decode()

    # read output
    glam_eval = dict()
    stdout_lines = stdout.split("\n")
    for line in stdout_lines:
        if "=" in line:
            metric, val = re.split("=| ", line)[0:2]
            glam_eval[metric] = val

    if os.path.exists(tmp_G_file):
        os.remove(tmp_G_file)

    return glam_eval
