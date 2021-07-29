#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import argparse
import numpy as np
import networkx as nx
import subprocess
import os
from tulip import tlp
import pandas as pd
from evalne.utils import preprocess as pp

def parse_args():
    """ Parses Fruchterman-Reingold arguments."""

    parser = argparse.ArgumentParser(
        description="Run force-based Fruchterman-Reingold algorithm."
    )

    parser.add_argument("--inputgraph", nargs="?", help="Input graph path")

    parser.add_argument(
        "--output",
        nargs="?",
        default=None,
        help="Path where the embeddings will be stored.",
    )

    parser.add_argument(
        "--dimension", type=int, default=2, help="Embedding dimension. Default is 2."
    )

    parser.add_argument(
        "--epochs", type=int, default=100, help="Training epochs. Default is 100."
    )

    parser.add_argument(
        "--delimiter", default=",", help="The delimiter used to separate the edgelist."
    )

    parser.add_argument(
        "--mode",
        default="rtx",
        type=str,
        choices=["networkx", "rtx", "naive", "lbvh"],
        help="Choose FR implementation. Networkx = networkx.spring_layout and rtx as \
        well as lbvh and naive use the owl-graph-drawing implementation",
    )

    parser.add_argument(
        "--exec",
        type=str,
        default="methods/frrtx/build/gd",
        help="Path where the executable 'gd' can be found.",
    )
    return parser.parse_args()


def tulip2txt(input, output, delimiter=","):
    """
    https://tulip.labri.fr/Documentation/current/tulip-python/html/tulippluginsdocumentation.html#csv-export
    """

    tmp_tlp_output = os.path.join(os.path.split(output)[0], "tmp_tlp_output.csv")

    # read tlp file
    params = tlp.getDefaultPluginParameters("TLP Import")
    params["filename"] = input
    graph = tlp.importGraph("TLP Import", params)

    # export layout properties to csv
    params = tlp.getDefaultPluginParameters("CSV Export", graph)
    params["Field separator"] = "Custom"
    params["Custom separator"] = ";"
    params["Type of elements"] = "nodes"
    params["Export id"] = True
    params["Export visual properties"] = True
    tlp.exportGraph("CSV Export", graph, tmp_tlp_output, params)

    # read tmp export and extract coordinates
    tlp_graph = pd.read_table(tmp_tlp_output, delimiter=";")
    coordinates = tlp_graph["viewLayout"].values
    coordinates = [
        [float(tp.strip("()").split(",")[0]), float(tp.strip("()").split(",")[1])]
        for tp in coordinates
    ]
    np.savetxt(output, coordinates, delimiter=delimiter)

    # delete temporary file
    if os.path.isfile(tmp_tlp_output):
        os.remove(tmp_tlp_output)


def tulip2edgestxt(input, output, delimiter=","):
    """
    https://tulip.labri.fr/Documentation/current/tulip-python/html/tulippluginsdocumentation.html#csv-export
    """

    tmp_tlp_output = os.path.join(os.path.split(output)[0], "tmp_tlp_output.csv")

    # read tlp file
    params = tlp.getDefaultPluginParameters("TLP Import")
    params["filename"] = input
    graph = tlp.importGraph("TLP Import", params)

    # export layout properties to csv
    params = tlp.getDefaultPluginParameters("CSV Export", graph)
    params["Field separator"] = "Custom"
    params["Custom separator"] = ";"
    params["Type of elements"] = "edges"
    params["Export id"] = True
    params["Export visual properties"] = False
    tlp.exportGraph("CSV Export", graph, tmp_tlp_output, params)

    # read tmp export and extract coordinates
    tlp_graph = pd.read_table(tmp_tlp_output, delimiter=";")
    src = tlp_graph["src id"].values
    target = tlp_graph["tgt id"].values
    np.savetxt(output, np.transpose(np.array([src, target])), delimiter=delimiter, fmt="%i")

    # delete temporary file
    if os.path.isfile(tmp_tlp_output):
        os.remove(tmp_tlp_output)


def run_RT_FR(input_file, output_file, delimiter, epochs, exec, mode):
    # the implementation needs a header for the edgelist input file
    edges = np.loadtxt(input_file, delimiter=",", dtype=int)

    # transform file into csv with header
    tmp_rtx_edgefle = os.path.join(os.path.split(input_file)[0], "tmp_rtx_edgefile.csv")
    np.savetxt(tmp_rtx_edgefle, edges, delimiter=",", fmt="%i", header="edgelist")

    tmp_tulip_output = os.path.join(os.path.split(output_file)[0], "tmp_tulip.tlp")

    start = time.time()
    process = subprocess.Popen(
        [
            exec,
            "-bench=true",
            "-dt=file",
            "-mode=" + mode,
            "-n=" + str(epochs),
            "-o=" + tmp_tulip_output,
            tmp_rtx_edgefle,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf8"
    )

    while True:
        if process.poll() is not None:
            break
        output = process.stdout.readline()
        if output:
            dec_output = output.strip()
            if "done" in dec_output:
                break
    
    runtime = time.time() - start
    print(f"Embedding time: {runtime:.4f}")

    assert os.path.isfile(tmp_tulip_output), f"FR-RTX did not produce the output file {tmp_tulip_output}."
    process.terminate()

    # convert tlp into txt embedding
    tulip2txt(tmp_tulip_output, output_file, delimiter)

    # write out edges from tulip
    tmp_tulip_edgefile = "tmp_tlp_edges.txt"
    tulip2edgestxt(tmp_tulip_output, tmp_tulip_edgefile, delimiter)
    tlp_edges = np.loadtxt(tmp_tulip_edgefile, delimiter=",", dtype=int)

    embedding = np.loadtxt(output_file, delimiter=delimiter)

    if not (edges == tlp_edges).all():
        print("Nodes are relabeled by FR-RTX. Computing node mapping.")
        # edgelists are not the same -- so node IDs should be in different order
        flat_edges = edges.flatten()
        indexes = np.unique(edges.flatten(), return_index=True)[1]
        unique_edges = [flat_edges[index] for index in sorted(indexes)]
        node_order = np.argsort(unique_edges)
        embedding = embedding[node_order]

    if os.path.isfile(tmp_tulip_output):
        os.remove(tmp_tulip_output)
    if os.path.isfile(tmp_tulip_edgefile):
        os.remove(tmp_tulip_edgefile)
    if os.path.isfile(tmp_rtx_edgefle):
        os.remove(tmp_rtx_edgefle)

    np.savetxt(output_file, embedding, delimiter=delimiter)

    return float(runtime)


def spring_layout(input_file, output_file, delimiter, epochs):
    G = pp.load_graph(input_file, delimiter=delimiter)
    G, _ = pp.prep_graph(G)

    # Compute Fruchterman-Reingold layout
    start = time.time()
    embedding_dict = nx.spring_layout(G, iterations=epochs)
    runtime = time.time() - start
    print(f"Embedding time: {runtime:.4f}")
    embedding_dict = dict(sorted(embedding_dict.items(), key=lambda item: item[0]))
    embedding = np.array(list(embedding_dict.values()))

    # Store the embedding in output file
    np.savetxt(output_file, embedding, delimiter=delimiter)


def main(args):
    # Naive spring layout implementation
    if args.mode == "networkx" or not os.path.isfile(args.exec):
        spring_layout(args.inputgraph, args.output, args.delimiter, args.epochs)

    # Use faster owl-graph-drawing implementation
    else:
        run_RT_FR(
            args.inputgraph,
            args.output,
            args.delimiter,
            args.epochs,
            args.exec,
            args.mode,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)