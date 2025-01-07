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

from abc import ABC
from abc import abstractmethod

import tensorflow_datasets as tfds

import nncf
from examples.tensorflow.common.logger import logger
from examples.tensorflow.common.utils import set_hard_limit_num_open_files


class BaseDatasetBuilder(ABC):
    """Abstract dataset loader and input processing."""

    def __init__(self, config, is_train, num_devices):
        self._config = config

        self._is_train = is_train
        self._num_devices = num_devices
        self._global_batch_size = config.batch_size

        # Dataset params
        self._dataset_dir = config.dataset_dir
        self._dataset_name = config.get("dataset", None)
        self._dataset_type = config.get("dataset_type", "tfds")
        self._as_supervised = False

        # Dataset loader
        self._dataset_loader = None

        # TFDS params
        self._skip_decoding = False

        # Dict with TFRecordDatasets
        self._tfrecord_datasets = {}

        self._split = "train" if self._is_train else "validation"

    @property
    def is_train(self):
        """Returns a `bool` flag which specifies whether it is a training or evaluation dataset."""
        return self._is_train

    @property
    def batch_size(self):
        """Returns per replica batch size."""
        return self._global_batch_size // self._num_devices

    @property
    def global_batch_size(self):
        """Returns global batch size."""
        return self.batch_size * self._num_devices

    @property
    def steps_per_epoch(self):
        """Returns steps per epoch"""
        return self.num_examples // self.global_batch_size

    @property
    @abstractmethod
    def num_examples(self):
        """Returns number of examples in the current dataset."""

    @property
    @abstractmethod
    def num_classes(self):
        """Returns number of classes in the current dataset."""

    @abstractmethod
    def _pipeline(self, dataset):
        """The pipeline which decodes and preprocesses the input data for model."""

    def build(self):
        dataset_builders = {
            "tfds": self._load_tfds,
            "tfrecords": self._load_tfrecords,
        }

        builder = dataset_builders.get(self._dataset_type, None)
        if builder is None:
            raise nncf.UnknownDatasetError("Unknown dataset type {}".format(self._dataset_type))

        dataset = builder()
        dataset = self._pipeline(dataset)

        return dataset

    def _load_tfds(self):
        logger.info("Using TFDS to load {} data.".format(self._split))

        set_hard_limit_num_open_files()

        self._dataset_loader = tfds.builder(self._dataset_name, data_dir=self._dataset_dir)

        self._dataset_loader.download_and_prepare()

        decoders = {"image": tfds.decode.SkipDecoding()} if self._skip_decoding else None

        read_config = tfds.ReadConfig(interleave_cycle_length=64, interleave_block_length=1)

        dataset = self._dataset_loader.as_dataset(
            split=self._split,
            as_supervised=self._as_supervised,
            shuffle_files=self._is_train,
            decoders=decoders,
            read_config=read_config,
        )

        return dataset

    def _load_tfrecords(self):
        logger.info("Using TFRecords to load {} data.".format(self._split))

        dataset_key = self._dataset_name.replace("/", "")
        if dataset_key in self._tfrecord_datasets:
            self._dataset_loader = self._tfrecord_datasets[dataset_key](config=self._config, is_train=self._is_train)
        else:
            raise nncf.UnknownDatasetError("Unknown dataset name: {}".format(self._dataset_name))

        dataset = self._dataset_loader.as_dataset()

        return dataset
