# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import pdb
import sys
import tarfile
from os import path as osp
from pathlib import Path
from shutil import copyfile
from typing import Tuple

import torch
from PIL import Image
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from examples.common.sample_config import SampleConfig
from examples.torch.common.distributed import configure_distributed
from examples.torch.common.example_logger import logger as default_logger
from examples.torch.common.execution import ExecutionMode
from examples.torch.common.execution import get_device
from nncf.config.schemata.defaults import QUANTIZATION_BITS

GENERAL_LOG_FILE_NAME = "output.log"
NNCF_LOG_FILE_NAME = "nncf_output.log"


def get_run_name(config: SampleConfig) -> str:
    dataset = config.get("dataset", "imagenet")
    if dataset is None:
        dataset = "imagenet"
    retval = config["model"] + "_" + dataset
    compression_config = config.get("compression", [])
    if not isinstance(compression_config, list):
        compression_config = [
            compression_config,
        ]
    for algo_dict in compression_config:
        algo_name = algo_dict["algorithm"]
        if algo_name == "quantization":
            initializer = algo_dict.get("initializer", {})
            precision = initializer.get("precision", {})
            if precision:
                retval += "_mixed_int"
            else:
                activations = algo_dict.get("activations", {})
                a_bits = activations.get("bits", QUANTIZATION_BITS)
                weights = algo_dict.get("weights", {})
                w_bits = weights.get("bits", QUANTIZATION_BITS)
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
    with open(filename, "w", encoding="utf8") as outfile:
        json.dump(metrics, outfile)


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

    training_pipeline_log_file_handler = logging.FileHandler(
        str(Path(config.log_dir) / GENERAL_LOG_FILE_NAME), encoding="utf-8"
    )
    training_pipeline_log_file_handler.setFormatter(logging.Formatter("%(message)s"))
    sample_logger.addHandler(training_pipeline_log_file_handler)

    nncf_log_file_handler = logging.FileHandler(str(Path(config.log_dir) / NNCF_LOG_FILE_NAME), encoding="utf-8")
    nncf_log_file_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    from nncf.common.logging import nncf_logger

    nncf_logger.addHandler(nncf_log_file_handler)


def is_on_first_rank(config):
    return not config.multiprocessing_distributed or (
        config.multiprocessing_distributed and config.rank % config.ngpus_per_node == 0
    )


def create_code_snapshot(root, dst_path, extensions=(".py", ".json", ".cpp", ".cu", ".h", ".cuh")):
    """Creates tarball with the source code"""
    with tarfile.open(str(dst_path), "w:gz") as tar:
        for path in Path(root).rglob("*"):
            if ".git" in path.parts:
                continue
            if path.suffix.lower() in extensions:
                tar.add(path.as_posix(), arcname=path.relative_to(root).as_posix(), recursive=True)


def print_args(config, logger=default_logger):
    logger.info("\nConfiguration parameters:")
    for arg in sorted(config):
        logger.info("{: <27s}: {}".format(arg, config.get(arg)))
    logger.info("\n")


def make_link(src, dst, exists_ok=True):
    if osp.exists(dst) and exists_ok:
        os.remove(dst)
    dev1 = os.stat(osp.dirname(dst)).st_dev
    dev2 = os.stat(src).st_dev
    if dev1 != dev2:
        copyfile(src, dst)
    else:
        os.link(src, dst)


def make_additional_checkpoints(checkpoint_path: str, is_best: bool, epoch: int, config: SampleConfig):
    if is_best:
        best_path = Path(config.checkpoint_save_dir) / f"{config.name}_best.pth"
        copyfile(checkpoint_path, str(best_path))
    if epoch % config.save_freq == 0:
        intermediate_checkpoint = Path(config.intermediate_checkpoints_path) / f"epoch_{epoch}.pth"
        copyfile(checkpoint_path, str(intermediate_checkpoint))


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            with open("/dev/stdin", encoding="utf8") as file:
                sys.stdin = file
                pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def is_staged_quantization(config):
    compression_config = config.get("compression", {})
    if isinstance(compression_config, list):
        compression_config = compression_config[0]
    algo_type = compression_config.get("algorithm")
    if algo_type is not None and algo_type == "quantization" and compression_config.get("params", {}):
        return True
    return False


def is_pretrained_model_requested(config: SampleConfig) -> bool:
    return config.get("pretrained", True) if config.get("weights") is None else False


class MockDataset(data.Dataset):
    def __init__(self, img_size: Tuple[int, int] = (32, 32), num_images: int = 1000, transform=None):
        super().__init__()
        self._img_size = img_size
        self._num_images = num_images
        self._transform = transform

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        if 0 <= idx < self._num_images:
            img = Image.new(mode="RGB", size=self._img_size)
            if self._transform is not None:
                img = self._transform(img)
            return img, 0
        raise ValueError


class NullContextManager:
    """
    Dummy context manager that do nothing.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass
