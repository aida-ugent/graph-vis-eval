#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import random
from datetime import datetime
from configparser import ConfigParser, ExtendedInterpolation
import numpy as np
import graph_embedding_lp_evaluator as gev
import config_utils

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation on link prediction task.")
    parser.add_argument("config", default="evalne_config.ini", nargs='?',
                        help="Config file")
    return parser.parse_args()


def main(args):
    # Read the configuration file
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(args.config)
    
    load_traintest_split = config.get("SETUP",
                                      "load_traintest_split",
                                      fallback=None)
    if load_traintest_split is not None:
        if not os.path.exists(load_traintest_split):
            raise ValueError(f"Path {load_traintest_split} does not exist.\
                Cannot load train/test split.")

    # Check that output paths exist and create folders if necessary
    output = config.get("SETUP", "output")

    if not os.path.isdir(output):
        os.mkdir(output)

    # Create new folder for experiments
    today = datetime.now()
    new_output = output + "/" + today.strftime("%Y%m%d%H%M%S") + "_evalne"
    os.mkdir(new_output)
    config.set("SETUP", "output", new_output)
    
    # copy config to directory
    with open(new_output + "/" + "evalne_config.ini", "w") as fp:
        config.write(fp)

    GE = gev.GraphLPEval(config_utils.get_dataset_dict(config),
                              config_utils.get_methods_dict(config),
                              config.get("SETUP", "metrics").split(),
                              new_output,
                              config.getint("SETUP", "repetitions"),
                              emb_dimension=config.getint("SETUP", "emb_dimension", fallback=2),
                              load_traintest_split=load_traintest_split,
                              baselines=config.get("SETUP", "baselines").split(),
                              edge_embed_methods = config.get("SETUP", "edge_embed_methods").split())
    GE.evaluate_all()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    args = parse_args()
    main(args)
