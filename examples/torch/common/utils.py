"""
 Copyright (c) 2019 Intel Corporation
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

import datetime
import json
import logging
import pdb
import sys
from os import path as osp
from pathlib import Path

import os
import tarfile
from shutil import copyfile
from typing import Tuple

from PIL import Image
import torch.utils.data as data

from examples.torch.common.distributed import configure_distributed
from examples.torch.common.execution import ExecutionMode, get_device
from examples.torch.common.sample_config import SampleConfig
from torch.utils.tensorboard import SummaryWriter
import mlflow
import torch

from examples.torch.common.example_logger import logger as default_logger
from nncf.torch.utils import is_main_process

# pylint: disable=import-error
from returns.maybe import Maybe, Nothing

GENERAL_LOG_FILE_NAME = "output.log"
NNCF_LOG_FILE_NAME = "nncf_output.log"


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

    if config.nncf_config is not None:
        config.nncf_config["log_dir"] = config.log_dir

    if config.checkpoint_save_dir is None:
        config.checkpoint_save_dir = config.log_dir

    # create aux dirs
    config.intermediate_checkpoints_path = config.log_dir + '/intermediate_checkpoints'
    os.makedirs(config.intermediate_checkpoints_path)
    os.makedirs(config.checkpoint_save_dir, exist_ok=True)


class SafeMLFLow:
    """ Wrapper to encapsulate safe access to mlflow methods without checking for None and with automatic closing """

    def __init__(self, config):
        self.is_suitable_mode = config.mode.lower() == 'train' and config.to_onnx is None
        root_log_dir = osp.dirname(osp.dirname(config.log_dir))
        self.safe_call('set_tracking_uri', osp.join(root_log_dir, 'mlruns'))

        self.safe_call('get_experiment_by_name', config.name). \
            or_else_call(lambda: self.safe_call('create_experiment', config.name))

        self.safe_call('set_experiment', config.name)
        self.safe_call('start_run')

        def create_symlink_fn(mlflow_active_run: mlflow.ActiveRun):
            os.symlink(config.log_dir, osp.join(mlflow_active_run.info.artifact_uri, osp.basename(config.log_dir)))
            return Nothing

        self.safe_call('active_run').bind(create_symlink_fn)

    def __del__(self):
        self.safe_call('end_run')

    def safe_call(self, func: str, *args, **kwargs) -> Maybe:
        """ Calls mlflow method, if it's enabled and safely does nothing in the opposite case"""
        return Maybe.from_value(self._get_mlflow()).bind(
            lambda obj: Maybe.from_value(getattr(obj, func)(*args, **kwargs)))

    def end_run(self):
        self.safe_call('end_run')

    def _is_enabled(self):
        return self.is_suitable_mode and is_main_process()

    def _get_mlflow(self):
        if self._is_enabled():
            return mlflow
        return None


def configure_device(current_gpu, config: SampleConfig):
    config.current_gpu = current_gpu
    config.distributed = config.execution_mode in (ExecutionMode.DISTRIBUTED, ExecutionMode.MULTIPROCESSING_DISTRIBUTED)
    if config.distributed:
        configure_distributed(config)

    config.device = get_device(config)

    if config.execution_mode == ExecutionMode.SINGLE_GPU:
        torch.cuda.set_device(config.current_gpu)


def configure_logging(sample_logger, config):
    config.tb = SummaryWriter(config.log_dir)

    training_pipeline_log_file_handler = logging.FileHandler(osp.join(config.log_dir, GENERAL_LOG_FILE_NAME))
    training_pipeline_log_file_handler.setFormatter(logging.Formatter("%(message)s"))
    sample_logger.addHandler(training_pipeline_log_file_handler)

    nncf_log_file_handler = logging.FileHandler(osp.join(config.log_dir, NNCF_LOG_FILE_NAME))
    nncf_log_file_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    from nncf.common.utils.logger import logger as nncf_logger
    nncf_logger.addHandler(nncf_log_file_handler)


def log_common_mlflow_params(config):
    optimizer = config.nncf_config.get('optimizer', {})
    config.mlflow.safe_call('log_param', 'epochs', config.get('epochs', 'None'))
    config.mlflow.safe_call('log_param', 'schedule_type', optimizer.get('schedule_type', 'None'))
    config.mlflow.safe_call('log_param', 'lr', optimizer.get('base_lr', 'None'))
    config.mlflow.safe_call('set_tag', 'Log Dir Path', config.log_dir)


def is_on_first_rank(config):
    return not config.multiprocessing_distributed or (config.multiprocessing_distributed
                                                      and config.rank % config.ngpus_per_node == 0)


def create_code_snapshot(root, dst_path, extensions=(".py", ".json", ".cpp", ".cu", ".h", ".cuh")):
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


def make_link(src, dst, exists_ok=True):
    if osp.exists(dst) and exists_ok:
        os.remove(dst)
    dev1 = os.stat(osp.dirname(dst)).st_dev
    dev2 = os.stat(src).st_dev
    if dev1 != dev2:
        copyfile(src, dst)
    else:
        os.link(src, dst)


def make_additional_checkpoints(checkpoint_path, is_best, epoch, config):
    if is_best:
        best_path = osp.join(config.checkpoint_save_dir, '{}_best.pth'.format(config.name))
        copyfile(checkpoint_path, best_path)
    if epoch % config.save_freq == 0:
        intermediate_checkpoint = osp.join(config.intermediate_checkpoints_path,
                                           'epoch_{}.pth'.format(epoch))
        copyfile(checkpoint_path, intermediate_checkpoint)


# pylint:disable=no-member
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def is_staged_quantization(config):
    compression_config = config.get('compression', {})
    if isinstance(compression_config, list):
        compression_config = compression_config[0]
    algo_type = compression_config.get("algorithm")
    if algo_type is not None and algo_type == "binarization":
        return True
    if algo_type == "quantization" and compression_config.get("params", {}):
        return True
    return False


def is_pretrained_model_requested(config: SampleConfig) -> bool:
    return config.get('pretrained', True) if config.get('weights') is None else False


class MockDataset(data.Dataset):
    def __init__(self, img_size: Tuple[int, int] = (32, 32), num_images: int = 1000,
                 transform=None):
        super().__init__()
        self._img_size = img_size
        self._num_images = num_images
        self._transform = transform

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        if 0 <= idx < self._num_images:
            img = Image.new(mode='RGB', size=self._img_size)
            if self._transform is not None:
                img = self._transform(img)
            return img, 0
        raise ValueError
