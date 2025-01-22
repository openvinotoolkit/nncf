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

from functools import partial

import tensorflow as tf

from examples.tensorflow.common.dataset_builder import BaseDatasetBuilder
from examples.tensorflow.common.object_detection.datasets import tfrecords as records_dataset
from examples.tensorflow.common.object_detection.datasets.preprocessing_selector import get_preprocess_input_fn


class COCODatasetBuilder(BaseDatasetBuilder):
    """COCO2017 dataset loader and input processing."""

    def __init__(self, config, is_train, num_devices):
        super().__init__(config, is_train, num_devices)

        # Pipeline params
        self._shuffle_buffer_size = 1000
        self._num_preprocess_workers = config.get("workers", tf.data.experimental.AUTOTUNE)
        self._cache = False
        self._include_mask = config.get("include_mask", False)

        # TFDS params
        self._skip_decoding = True

        self._tfrecord_datasets = records_dataset.__dict__

    @property
    def num_examples(self):
        if self._dataset_type == "tfds":
            return self._dataset_loader.info.splits[self._split].num_examples
        if self._dataset_type == "tfrecords":
            return self._dataset_loader.num_examples
        return None

    @property
    def num_classes(self):
        if self._dataset_type == "tfds":
            return self._dataset_loader.info.features["objects"]["label"].num_classes
        if self._dataset_type == "tfrecords":
            return self._dataset_loader.num_classes
        return None

    def _pipeline(self, dataset):
        if self._cache:
            dataset = dataset.cache()

        if self._is_train:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(self._shuffle_buffer_size)

        tfds_decoder, preprocess_input_fn = get_preprocess_input_fn(self._config, self._is_train)

        if self._dataset_type == "tfrecords":
            decoder_fn = partial(
                self._dataset_loader.decoder,
                include_mask=self._include_mask,
                model=self._config.model,
                is_train=self._is_train,
            )
        else:
            decoder_fn = tfds_decoder

        dataset = preprocess_input_fn(dataset, decoder_fn)

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
