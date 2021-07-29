#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from configparser import ConfigParser, ExtendedInterpolation
import argparse
import random
import numpy as np
from datetime import datetime
from evalne.utils import preprocess as pp
import graph_embedding_evaluator as gev
import config_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Run CNE.")
    parser.add_argument('--config', default='config.ini',
                        help='Config file')
    return parser.parse_args()


def main(args):
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(args.config)

    load_embeddings = config.get('SETUP', 'load_embeddings', fallback=None)
    if load_embeddings is not None:
        if not os.path.exists(load_embeddings):
            raise ValueError(
                f"Path {load_embeddings} does not exist. Cannot load embeddings.")

    # Check that output paths exist and create folders if necessary
    output = config.get('SETUP', 'output')
    if not os.path.isdir(output):
        os.mkdir(output)

    # Create new folder for experiments
    today = datetime.now()
    new_output = os.path.join(output, today.strftime('%Y%m%d%H%M%S'))
    os.mkdir(new_output)
    config.set('SETUP', 'output', new_output)

    # copy config to directory
    with open(os.path.join(new_output, "config.ini"), 'w') as fp:
        config.write(fp)

    GE = gev.GraphEval(datasets=config_utils.get_dataset_dict(config),
                       methods=config_utils.get_methods_dict(config),
                       metrics=config_utils.get_metrics_dict(config),
                       output_path=new_output,
                       repetitions=config.getint(
                           'SETUP', 'repetitions', fallback=1),
                       emb_dimension=config.getint(
                           'SETUP', 'emb_dimension', fallback=2),
                       visualize=config.getboolean(
                           'SETUP', 'visualize', fallback=True),
                       load_embeddings=load_embeddings)

    GE.evaluate_all()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    args = parse_args()
    main(args)
