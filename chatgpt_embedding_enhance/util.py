import logging
import os
import sys
import torch
import numpy as np
import argparse
import re
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, roc_curve, accuracy_score, matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr
from collections import Counter
import click
import functools




def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")





class NullLogger:
    def log_metric(self, *args, **kwargs):
        pass

    def log_metrics(self, *args, **kwargs):
        pass

    def log_parameter(self, *args, **kwargs):
        pass

    def log_parameters(self, *args, **kwargs):
        pass



class CometLogger:
    def __init__(self, args):
        import comet_ml

        comet_args = dict(
            project_name=args.comet_project_name,
            auto_metric_logging=args.comet_auto_metric_logging,
            auto_output_logging=args.comet_auto_output_logging,
            log_code=args.comet_log_code,
            log_env_cpu=args.comet_log_env_cpu,
            log_env_gpu=args.comet_log_env_gpu,
            log_env_host=args.comet_log_env_host,
            log_graph=args.comet_log_graph,
        )
        if args.comet_offline:
            self.logger = comet_ml.OfflineExperiment(offline_directory=args.comet_offline_dir, **comet_args)
        else:
            self.logger = comet_ml.Experiment(**comet_args)

    def log_metric(self, *args, **kwargs):
        self.logger.log_metric(*args, **kwargs)

    def log_metrics(self, *args, **kwargs):
        self.logger.log_metrics(*args, **kwargs)

    def log_parameter(self, *args, **kwargs):
        self.logger.log_parameter(*args, **kwargs)

    def log_parameters(self, *args, **kwargs):
        self.logger.log_parameters(*args, **kwargs)