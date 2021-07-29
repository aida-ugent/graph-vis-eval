#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import argparse
import numpy as np
import subprocess
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run DRGraph algorithm.")

    parser.add_argument("--inputgraph",
                        help="Input graph path")

    parser.add_argument("--output",
                        help="Path where the embeddings will be stored.")

    parser.add_argument("--dimension", type=int, default=2,
                        help="Embedding dimension. Default is 2.")

    parser.add_argument("--delimiter", default=",",
                        help="The delimiter used to separate the edgelist.")

    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Weight of negative samples. A small gamma leads to small repulsive forces.")
    
    parser.add_argument("--epochs", type=int, default=400,
                        help="Number of iterations.")
    
    parser.add_argument("--neg_samples", type=int, default=5,
                        help="Number of negative samples.")
    
    parser.add_argument("--mode", type=int, default=1, choices=[0, 1],
                        help="Algorithm mode 0: dimension reduction, 1: graph layout")
    
    parser.add_argument("--A", type=float, default=2.0,
                        help="??? not described in paper or code")
    
    parser.add_argument("--B", type=float, default=1.0,
                        help="Affecting the sum of the forces - small B causes dense local clusters,\
                            large B tries to preserva all distances.")

    parser.add_argument("--exec", type=str, default="methods/drgraph/Vis",
                        help="Path where the executable 'Vis' can be found.")
    return parser.parse_args()


def run_DRGraph(input_file, output_file, delimiter, exec,
                neg_samples, epochs, gamma, mode, A, B):
    
    # add weight of 1 to each edge
    edges = np.loadtxt(input_file, delimiter=",", dtype=int)
    edges = np.hstack((edges, np.ones((edges.shape[0],1))))

    # transform file into csv with header
    tmp_edgefile = os.path.join(os.path.split(input_file)[0], "tmp_edgefile.txt")
    tmp_output_file = os.path.join(os.path.split(output_file)[0], "tmp_output.txt")

    np.savetxt(tmp_edgefile, edges, delimiter=" ", fmt="%i",
               comments="",
               header=str(int(np.amax(edges)) + 1) + " " + str(edges.shape[0]))

    cmd = [
            exec,
            "-input", tmp_edgefile,
            "-output", tmp_output_file,
            "-neg", str(neg_samples),
            "-samples", str(epochs),
            "-gamma", str(gamma),
            "-mode", str(mode),
            "-A", str(A),
            "-B", str(B)
        ]
    
    runtime = 0.0
    try:
        start = time.time()
        output = subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
                encoding='utf8',
            )
        runtime = time.time() - start
        print(f"Embedding time: {runtime:.4f}")
    except subprocess.CalledProcessError as e:
        print(f"Calling DRGraph failed with following output:\n{e.output}")
        exit(1)

    # change delimiter
    emb = np.loadtxt(tmp_output_file, dtype=float, delimiter=" ", skiprows=1)
    with open(output_file, 'wb') as f:
        np.savetxt(f, emb, delimiter=delimiter)
        os.fsync(f)

    if os.path.isfile(tmp_output_file):
        os.remove(tmp_output_file)
    if os.path.isfile(tmp_edgefile):
        os.remove(tmp_edgefile)


def main(args):
    run_DRGraph(
        args.inputgraph,
        args.output,
        args.delimiter,
        exec=args.exec,
        neg_samples=args.neg_samples,
        epochs=args.epochs,
        gamma=args.gamma,
        mode=args.mode,
        A=args.A,
        B=args.B
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)