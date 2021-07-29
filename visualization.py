#!/usr/bin/env python
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm


def draw_edges(G, emb, filename=None, cmap='viridis', format="png"):
    # could also use 'jet_r' colormap
    fig = plt.figure(figsize=(5, 5))
    plt.axis('off')
    nx.draw_networkx(G, emb, node_color='white',
                     node_size=0,
                     alpha=1,
                     width=0.4,
                     with_labels=False,
                     edge_color=edge_length_colors(G, emb, cmap))
    if filename is not None:
        fig.savefig(filename + "." + format, dpi=300,
                    format=format, bbox_inches="tight", pad_inches=0)
    plt.close('all')


def draw_nodes(G, emb, filename=None, cmap='viridis', format="png"):
    node_size = 25 / np.sqrt(len(G.nodes) / 50)
    edge_width = 0.4 if len(G) < 5000 else 0.2
    alpha = 0.2 if len(G) < 5000 else 0.1
    fig = plt.figure(figsize=(5, 5))
    plt.axis('off')

    nx.draw_networkx_nodes(G,
                     emb,
                     node_color=node_degree_colors(G, cmap),
                     node_size=node_size,
                     alpha=0.6,
                     width=edge_width,
                     with_labels=False,
                     edge_color=None,
                     linewidths=0)
    nx.draw_networkx_edges(G, emb, width=edge_width,
                           alpha=alpha, edge_color="gray")

    if filename is not None:
        fig.savefig(filename + "." + format, dpi=300,
                    format=format, bbox_inches="tight", pad_inches=0)
    plt.close('all')


def node_degree_colors(G, cmap):
    degrees = np.array(list(dict(G.degree()).values()))
    norm = matplotlib.colors.LogNorm(vmin=min(degrees), vmax=max(degrees))
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = mapper.to_rgba(degrees)
    return colors


def edge_length_colors(G, embedding, cmap):
    edge_lengths = np.array([np.linalg.norm(embedding[i] - embedding[j]) for i,j in G.edges()])
    mapper = cm.ScalarMappable(norm=matplotlib.colors.Normalize(), cmap=cmap)
    colors = mapper.to_rgba(edge_lengths)
    return colors
