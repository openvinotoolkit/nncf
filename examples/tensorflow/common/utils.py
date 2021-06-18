"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import time
import datetime
import json
import logging
import os
import tarfile
import resource
from os import path as osp
from pathlib import Path

import tensorflow as tf

from nncf.common.utils.logger import logger as nncf_logger
from examples.tensorflow.common.logger import logger as default_logger

GENERAL_LOG_FILE_NAME = "output.log"
NNCF_LOG_FILE_NAME = "nncf_output.log"

SAVED_MODEL_FORMAT = 'tf'
KERAS_H5_FORMAT = 'h5'
FROZEN_GRAPH_FORMAT = 'frozen_graph'


def get_name(config):
    dataset = config.get('dataset', 'imagenet')
    if dataset is None:
        dataset = 'imagenet'
    retval = config["model"] + "_" + dataset
    compression_config = config.get('compression', [])
    if not isinstance(compression_config, list):
        compression_config = [compression_config, ]
    for algo_dict in compression_config:
        algo_name = algo_dict["algorithm"]
        if algo_name == "quantization":
            initializer = algo_dict.get("initializer", {})
            precision = initializer.get("precision", {})
            if precision:
                retval += "_mixed_int"
            else:
                activations = algo_dict.get('activations', {})
                a_bits = activations.get('bits', 8)
                weights = algo_dict.get('weights', {})
                w_bits = weights.get('bits', 8)
                if a_bits == w_bits:
                    retval += "_int{}".format(a_bits)
                else:
                    retval += "_a_int{}_w_int{}".format(a_bits, w_bits)
        else:
            retval += "_{}".format(algo_name)
    return retval


def write_metrics(acc, filename):
    avg = round(acc * 100, 2)
    metrics = {"Accuracy": avg}
    with open(filename, 'w') as outfile:
        json.dump(metrics, outfile)


def configure_paths(config):
    d = datetime.datetime.now()
    run_id = '{:%Y-%m-%d__%H-%M-%S}'.format(d)
    config.name = get_name(config)
    config.log_dir = osp.join(config.log_dir, "{}/{}".format(config.name, run_id))
    os.makedirs(config.log_dir)

    compression_config = config.get('compression', [])
    if not isinstance(compression_config, list):
        compression_config = [compression_config, ]
    for algo_config in compression_config:
        algo_config.log_dir = config.log_dir

    if config.checkpoint_save_dir is None:
        config.checkpoint_save_dir = config.log_dir

    # create aux dirs
    os.makedirs(config.checkpoint_save_dir, exist_ok=True)


def configure_logging(sample_logger, config):
    training_pipeline_log_file_handler = logging.FileHandler(osp.join(config.log_dir, GENERAL_LOG_FILE_NAME))
    training_pipeline_log_file_handler.setFormatter(logging.Formatter("%(message)s"))
    sample_logger.addHandler(training_pipeline_log_file_handler)

    nncf_log_file_handler = logging.FileHandler(osp.join(config.log_dir, NNCF_LOG_FILE_NAME))
    nncf_log_file_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    nncf_logger.addHandler(nncf_log_file_handler)


def create_code_snapshot(root, dst_path, extensions=(".py", ".json", ".cpp", ".cu", "h", ".cuh")):
    """Creates tarball with the source code"""
    with tarfile.open(str(dst_path), "w:gz") as tar:
        for path in Path(root).rglob("*"):
            if '.git' in path.parts:
                continue
            if path.suffix.lower() in extensions:
                tar.add(path.as_posix(), arcname=path.relative_to(root).as_posix(), recursive=True)


def print_args(config, logger=default_logger):
    for arg in sorted(config):
        logger.info("{: <27s}: {}".format(arg, config.get(arg)))


def serialize_config(config, log_dir):
    with open(osp.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)


def get_saving_parameters(config):
    if config.to_frozen_graph is not None:
        return config.to_frozen_graph, FROZEN_GRAPH_FORMAT
    if config.to_saved_model is not None:
        return config.to_saved_model, SAVED_MODEL_FORMAT
    if config.to_h5 is not None:
        return config.to_h5, KERAS_H5_FORMAT
    save_path = os.path.join(config.log_dir, 'frozen_model.pb')
    return save_path, FROZEN_GRAPH_FORMAT


def set_hard_limit_num_open_files():
    _, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


class SummaryWriter:
    """Simple SummaryWriter for writing dictionary of metrics

    Attributes:
        writer: tf.SummaryWriter
    """

    def __init__(self, log_dir, name):
        """Inits SummaryWriter with paths

        Arguments:
            log_dir: the model folder path
            name: the summary subfolder name
        """
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, name)) # pylint: disable=E1101

    def __call__(self, metrics, step):
        """Write metrics to summary with the given writer

        Args:
            metrics: a dictionary of metrics values
            step: integer. The training step
        """

        with self.writer.as_default(): # pylint: disable=E1129
            for metric_name, value in metrics.items():
                tf.summary.scalar(metric_name, value, step=step)
        self.writer.flush()

    def close(self):
        self.writer.close()


class Timer:
    """A simple timer."""

    def __init__(self):
        self.reset()

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        return self.diff

    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
