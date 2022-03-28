"""
 Copyright (c) 2022 Intel Corporation
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

from abc import ABC, abstractmethod
import tensorflow as tf

# Loading specifications
BUFFER_SIZE = 8 * 1024 * 1024 # the number of bytes in the read buffer.
CYCLE_LENGTH = 16 # the number of input elements that will be processed concurrently
NUM_PARALLEL_CALLS = tf.data.experimental.AUTOTUNE # the number of threads for reading the dataset


class TFRecordDataset(ABC):
    def __init__(self, config, is_train):
        self.dataset_dir = config.dataset_dir
        self.is_train = is_train

        self.shuffle_train = config.get('seed') is None
        self.buffer_size = config.get('buffer_size', BUFFER_SIZE)
        self.cycle_length = config.get('cycle_length', CYCLE_LENGTH)
        self.num_parallel_calls = NUM_PARALLEL_CALLS

    @property
    @abstractmethod
    def num_examples(self):
        pass

    @property
    @abstractmethod
    def num_classes(self):
        pass

    @property
    @abstractmethod
    def file_pattern(self):
        pass

    @property
    @abstractmethod
    def decoder(self):
        pass

    def as_dataset(self) -> tf.data.Dataset:
        dataset = tf.data.Dataset.list_files(self.file_pattern, shuffle=(self.is_train and self.shuffle_train))

        dataset = dataset.interleave(
            lambda name: tf.data.TFRecordDataset(name, buffer_size=self.buffer_size),
            cycle_length=self.cycle_length,
            num_parallel_calls=self.num_parallel_calls)

        return dataset
