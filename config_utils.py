#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast


def get_dataset_dict(config):
    datasets = dict()
    
    for data_name in config.get("SETUP", "datasets").split():
        info = dict()
        info["name"] = data_name
        info["file"] = config.get(data_name, "file")
        if "graph_dist_file" in config[data_name]:
            info["graph_dist_file"] = config.get(data_name, "graph_dist_file")
        info["delimiter"] = config.get(data_name, "delimiter").strip("\'").replace(r"\s", " ")
        info["directed"] = config.getboolean(data_name, "directed", fallback=True)
        info["comments"] = config.get(data_name, "comments", fallback="#")
        datasets[data_name] = info
    return datasets


def get_metrics_dict(config):
    metrics = dict()
    
    for metric_name in config.get("SETUP", "metrics", fallback="").split():
        info = dict()
        info["method"] = config.get(metric_name, "method")
        info["args"] = ast.literal_eval(config.get(metric_name, "args", fallback=dict()))
        metrics[metric_name] = info        
    return metrics


def get_methods_dict(config):
    methods = dict()

    for method_name in config.get("SETUP", "methods").split():
        info = dict()
        info["name"] = method_name

        default_dimension = config.getint("SETUP", "dimensions")
        info["emb_dimension"] = config.getint(method_name, "emb_dimension", fallback=default_dimension)
        info["command"] = config.get(method_name, "command")
        info["parameters"] = ast.literal_eval(config.get(method_name, "parameters", fallback=""))
        info["type"] = config.get(method_name, "type", fallback=None)
        info["use_tsne"] = config.getboolean(method_name, "use_tsne", fallback=False)
        
        methods[method_name] = info
    return methods