#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import networkx as nx
from evalne.evaluation.evaluator import LPEvaluator
from evalne.evaluation.split import LPEvalSplit
from evalne.evaluation.score import Scoresheet
from evalne.utils import split_train_test as stt
from evalne.utils import preprocess as pp
from sklearn.manifold import TSNE
from evalne_postprocessing import GravisLPEvaluator


class GraphLPEval():
    def __init__(self, datasets, methods, metrics,
                 output_path, repetitions, emb_dimension=2,
                 load_traintest_split=None,
                 baselines=[],
                 edge_embed_methods=["hadamard", "weighted_l1",
                                     "weighted_l2", "average"]):
        self.datasets = datasets
        self.methods = methods
        self.metrics = metrics
        self.repetitions = repetitions
        self.output_path = output_path
        self.emb_dimension = emb_dimension
        self.load_traintest_split = load_traintest_split
        self.baselines = baselines
        self.edge_embed_methods = edge_embed_methods
        # Initialize result dataframe
        cols = ['dataset', 'method', 'rep', 'parameters']
        self.results = pd.DataFrame(data=None, columns=cols)

        self.store_columns = ["method", "dataset", "rep"] + self.metrics + ["parameters"]

    def generate_train_test_split(self, G, train_frac,
                                  output, split_id, nw_name):
        r"""
        Generating and storing train test splits using stt.


        Returns
        -------
        filenames : list
            A list of strings, the names given to the 4 files where the true and false train and test edge are stored.
        traintest_split: EvalSplit object
            EvalSplit object to instantiate LPEvaluator.
        """

        # Generate one train/test split with default parameters
        train_E, test_E = stt.split_train_test(G, train_frac=train_frac)

        # Compute set of false edges
        train_E_false, test_E_false = stt.generate_false_edges_cwa(
            G,
            train_E=train_E,
            test_E=test_E,
            num_fe_train=None,
            num_fe_test=None
        )

        traintest_split = LPEvalSplit()
        traintest_split.set_splits(
            train_E, train_E_false, test_E, test_E_false,
            directed=G.is_directed(), nw_name=nw_name,
            split_id=split_id, TG=G
        )

        # Store the computed edge sets to a file
        filenames = stt.store_train_test_splits(
            output,
            train_E=train_E,
            train_E_false=train_E_false,
            test_E=test_E,
            test_E_false=test_E_false,
            split_id=split_id
        )
        return traintest_split, filenames

    def update_results(self, repeat, dataset, scoresheet,
                       store_intermediate=False):
        """
        Extract results for specific repetition from scoresheet and
        write to file.
        """
        df_int = pd.DataFrame(data=None, columns=["method", "dataset", "rep"])
        for metric in self.metrics:
            df = scoresheet.get_pandas_df(metric, repeat=repeat)
            df["rep"] = repeat+1
            df["method"] = df.index
            df.reset_index(drop=True)
            df = pd.melt(
                df,
                id_vars=["method", "rep"],
                value_vars=dataset,
                value_name=metric,
                var_name="dataset",
            )
            df_int = df_int.merge(df,
                                  on=["method", "dataset", "rep"],
                                  how="outer")
        method_parameters = pd.DataFrame.from_dict(self.methods, orient="index")[["name", "parameters"]]
        method_parameters.rename(columns={'name': 'method'}, inplace=True)
        df_int = df_int.merge(method_parameters, on=["method"], how="left")
        self.results = pd.concat((self.results, df_int), sort=True)

        if store_intermediate:
            self.results\
                .to_csv(os.path.join(self.output_path, "lp_results_intermediate.txt"),
                        index=False,
                        float_format='%.4f',
                        columns=self.store_columns)

    def save_results(self):
        output = os.path.join(self.output_path, "lp_results.txt")
        self.results\
            .to_csv(output,
                    index=False,
                    float_format='%.4f',
                    columns=self.store_columns)
        print(f"Saved results to {output}.")
        intermediate_results = os.path.join(self.output_path, "lp_results_intermediate.txt")
        if os.path.isfile(intermediate_results):
            os.remove(intermediate_results)

    def evaluate_all(self):
        for data_name, data_dict in self.datasets.items():
            print(f"\n\nDataset {data_name}")
            data_dir = os.path.join(self.output_path, data_name)
            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)
            
            if self.load_traintest_split is None:
                traintest_path = os.path.join(data_dir, "traintest")
                trainvalid_path = os.path.join(data_dir, "trainvalid")
            else:
                traintest_path = os.path.join(self.load_traintest_split,
                                              data_name, "traintest")
                trainvalid_path = os.path.join(self.load_traintest_split,
                                               data_name, "trainvalid")
            data_dict["traintest_path"] = traintest_path
            data_dict["trainvalid_path"] = trainvalid_path
            self.preprocess_graph(data_dict)
            self.process_dataset(data_dict)
        self.save_results()

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
        
    def get_network_evaluator(self, data_dict, repeat):
        if self.load_traintest_split is not None:
            print("Reading traintest split from {}".format(data_dict["traintest_path"]))
            traintest_split = LPEvalSplit()
            (train_E, train_E_false,
             test_E, test_E_false) = pp.read_train_test(data_dict["traintest_path"], repeat)
            traintest_split.set_splits(
                train_E, train_E_false, test_E, 
                test_E_false, directed=data_dict["directed"], 
                nw_name=data_dict["name"], split_id=repeat
            )

            trainvalid_split = LPEvalSplit()
            (trainvalid_E, trainvalid_E_false,
             testvalid_E, testvalid_E_false) = pp.read_train_test(data_dict["trainvalid_path"], repeat)
            trainvalid_split.set_splits(
                trainvalid_E, trainvalid_E_false, testvalid_E, 
                testvalid_E_false, directed=data_dict["directed"], 
                nw_name=data_dict["name"], split_id=repeat
            )
        else:
            # make new train test valid split
            traintest_split, _ = self.generate_train_test_split(
                data_dict["G"], train_frac=0.8,
                output=data_dict["traintest_path"],
                split_id=repeat, nw_name=data_dict["name"]
            )

            trainvalid_split, _ = self.generate_train_test_split(
                traintest_split.TG, train_frac=0.9,
                output=data_dict["trainvalid_path"],
                split_id=repeat, nw_name=data_dict["name"]
            )
        #return LPEvaluator(traintest_split, trainvalid_split, dim=self.emb_dimension)
        return GravisLPEvaluator(traintest_split, trainvalid_split, dim=self.emb_dimension)
 
    def process_dataset(self, data_dict):
        scoresheet = Scoresheet(tr_te="test")

        for r in range(self.repetitions):
            data_name = data_dict["name"]
            nee = self.get_network_evaluator(data_dict, r)

            # Evaluate baselines
            for bl in self.baselines:
                result = nee.evaluate_baseline(method=bl)
                scoresheet.log_results(result)
                
            for method_name, method_dict in self.methods.items():
                print(f"\nMethod {method_name}, rep {r+1} of {self.repetitions}")

                command = method_dict["command"]
                if method_dict.get("parameters", "") != "":
                    command = command + " " + method_dict["parameters"]

                if method_dict.get("use_tsne", False):
                    postprocessing = lambda emb: TSNE(n_components=self.emb_dimension).fit_transform(emb)
                else:
                    postprocessing = None

                results = nee.evaluate_cmd(
                    method_name=method_name,
                    method_type=method_dict["type"],
                    command=command,
                    edge_embedding_methods=self.edge_embed_methods,
                    tune_params=data_dict.get("tune_params", None),
                    input_delim=",",
                    output_delim=",",
                    timeout=None,
                    verbose=True,
                    ne_postprocessing_fn=postprocessing,
                    embedding_dim=method_dict.get("emb_dimension")
                )
                # Log the list of results
                scoresheet.log_results(results)
            self.update_results(r, data_name, scoresheet,
                                store_intermediate=True)
