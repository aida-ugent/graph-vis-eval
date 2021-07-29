#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import subprocess
import evalmetrics
import networkx as nx
import time
from sklearn.manifold import TSNE
from subprocess import CalledProcessError, PIPE

from evalne.utils import preprocess as pp
import visualization


class GraphEval():
    def __init__(self, datasets, methods, metrics,
                 output_path, repetitions=1, emb_dimension=2,
                 visualize=True, load_embeddings=None):
        self.datasets = datasets
        self.methods = methods
        self.metrics = metrics
        self.repetitions = repetitions
        self.output_path = output_path
        self.emb_dimension = emb_dimension
        self.visualize = visualize
        # Initialize result dataframe
        cols = ['dataset', 'method', 'rep', 'parameters',
                'embedding', 'embedding_file', 'runtime']
        self.results = pd.DataFrame(data=None, columns=cols)

        metric_names = list(self.metrics.keys())
        if "glam" in metric_names:
            metric_names.remove("glam")
            metric_names += ["crosslessness", "edge_length_cv", "min_angle", "shape_gabriel"]
        self.store_columns = ["method", "dataset", "rep", "runtime"] + metric_names + ["parameters"]

        if load_embeddings is not None:
            self.emb_path = load_embeddings
        else:
            self.emb_path = output_path

    def update_results(self, rows, store_intermediate=False):
        self.results = self.results.append(rows,
                                           ignore_index=True)

        if store_intermediate:
            self.results\
                .to_csv(os.path.join(self.output_path, "results_intermediate.txt"),
                        index=False,
                        float_format='%.4f',
                        columns=self.store_columns,
                        header=f"Embeddings path: {self.emb_path}")

    def save_results(self):
        output = os.path.join(self.output_path, "results.txt")
        self.results\
            .to_csv(output,
                    index=False,
                    float_format='%.4f',
                    columns=self.store_columns,
                    header=f"Embeddings path: {self.emb_path}")
        print(f"Saved results to {output}.")
        intermediate_results = os.path.join(self.output_path, "results_intermediate.txt")
        if os.path.isfile(intermediate_results):
            os.remove(intermediate_results)

    def evaluate_all(self):
        for data_name, data_dict in self.datasets.items():
            print(f"\n\nDataset {data_name}")
            data_dir = os.path.join(self.output_path, data_name)
            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)
            self.load_graph(data_dict)
            self.process_dataset(data_name, data_dict)
        self.save_results()
        self.cleanup_edgefiles()

    def compute_metrics(self, data_dict, embedding):
        results = {}
        for metric_name, metric_dict in self.metrics.items():
            metric_method = getattr(evalmetrics, metric_dict["method"])
            res = metric_method(data_dict["G"], embedding,
                                **metric_dict["args"],
                                data_dict=data_dict)
            if metric_name == "glam":
                results.update(res)
            else:
                results[metric_name] = res
        return results
    
    def cleanup_edgefiles(self):
        for data_dict in self.datasets.values():
            if "edgefile" in data_dict:
                tmp_edgefile = data_dict["edgefile"]
                if os.path.exists(tmp_edgefile):
                    os.remove(tmp_edgefile)

    def compute_visualization(self, G, embedding, emb_prefix, format="pdf"):
        print(f"Visualizing results...")
        assert embedding.shape[1] <= 2, "Embeddings can only be visulized in two dimensions."
        visualization.draw_edges(G, embedding, filename=emb_prefix + "_edge_vis",
                                 format=format)
        visualization.draw_nodes(G, embedding, filename=emb_prefix + "_vis",
                                 format=format)

    def preprocess_graph(self, data_dict, edgelist_filepath=None):
        # Load and preprocess the network
        G = pp.load_graph(data_dict["file"],
                          delimiter=data_dict["delimiter"],
                          directed=data_dict["directed"])
        G, _ = pp.prep_graph(G)
        data_dict["G"] = G

        print("Number of nodes {}".format(len(G.nodes)))
        print("Number of edges {}".format(len(G.edges)))
        
        if edgelist_filepath is not None:
            data_dict["file"] = edgelist_filepath
            np.savetxt(os.path.join(edgelist_filepath, data_dict['name'] + ".txt"),
                       G.edges(), delimiter=",", fmt="%d", header=str(G.number_of_nodes()) + ' ' + str(len(G.edges())))

    def load_graph(self, data_dict):
        """
        Load graph from edgelist without preprocessing.
        """
        E = np.loadtxt(data_dict["file"],
                       delimiter=data_dict["delimiter"],
                       comments=data_dict["comments"], dtype=int)
        G = nx.Graph()
        G.add_edges_from(E)
        data_dict["G"] = G
        print("Number of nodes {}".format(len(G.nodes)))
        print("Number of edges {}".format(len(G.edges)))

    def process_dataset(self, data_name, data_dict):
        for method_name, method_dict in self.methods.items():
            emb_dir = os.path.join(self.emb_path, data_name, method_name)
            if not os.path.isdir(emb_dir):
                os.mkdir(emb_dir)
            
            parameters = method_dict["parameters"]
                
            for r in range(1, self.repetitions+1):
                print(
                    f"\nMethod {method_name}, rep {r} of {self.repetitions}")

                emb_prefix = (data_name
                                + "_" + method_name
                                + "_" + str(r))
                emb_filename = emb_prefix + "_embedding.txt"
                emb_filepath = os.path.join(emb_dir, emb_filename)
                if os.path.exists(emb_filepath):
                    print("Reading embedding from {}.".format(emb_filepath))
                    embedding = np.loadtxt(emb_filepath, delimiter=",")
                    exec_time = 0
                else:
                    command = (method_dict["command"]
                                + " --inputgraph " + data_dict["file"]
                                + " --output " + emb_filepath
                                + " --dimension " + str(method_dict["emb_dimension"]))
                    command = command + " " + parameters if parameters != "" else command
                    start = time.time()
                    
                    try:                        
                        subprocess.run(command, shell=True, stderr=PIPE, check=True, encoding='utf8')
                    except CalledProcessError as err:
                        print(f"Method call {command} failed with following output:\n{err.stderr}")
                        exit(1)
                    exec_time = time.time() - start
                    embedding = np.loadtxt(emb_filepath, delimiter=",")

                    if method_dict.get("use_tsne", False):
                        pretsne_filename = emb_filepath[:-4] + "_dim" + str(method_dict["emb_dimension"]) + ".txt"
                        os.rename(emb_filepath, pretsne_filename)
                        embedding = np.loadtxt(pretsne_filename, delimiter=",")
                        print(f"Using t-SNE to reduce dimensionality from {method_dict['emb_dimension']} to {self.emb_dimension}")
                        tsne_start = time.time()
                        embedding = TSNE(n_components=self.emb_dimension).fit_transform(embedding)
                        exec_time += time.time() - tsne_start
                        np.savetxt(emb_filepath, embedding, delimiter=",")

                # Check that embeddings all have final dimension
                assert embedding.shape[1] == self.emb_dimension, f"Embedding is expected to have {self.emb_dimension} dimensions."    
                    
                res = {'rep': r,
                        'dataset': data_name,
                        'method': method_name,
                        'embedding': embedding,
                        'embedding_file': os.path.join(data_name, method_name, emb_filename),
                        'runtime': exec_time,
                        'parameters': parameters,
                        'dimension': method_dict["emb_dimension"]}

                metric_results = self.compute_metrics(data_dict, embedding)
                res.update(metric_results)
                self.update_results(res, store_intermediate=True)

                if self.visualize and r == 1:
                    self.compute_visualization(data_dict["G"],
                                            embedding,
                                            os.path.join(self.output_path, data_name, emb_prefix))
