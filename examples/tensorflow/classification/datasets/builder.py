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

import tensorflow as tf

from examples.tensorflow.common.dataset_builder import BaseDatasetBuilder
from examples.tensorflow.classification.datasets import tfrecords as records_dataset
from examples.tensorflow.classification.datasets.preprocessing_selector import get_preprocessing
from examples.tensorflow.classification.datasets.preprocessing_selector import get_label_preprocessing_fn

DATASET_SPLITS = {
    'imagenet2012': ('train', 'validation'),
    'cifar10': ('train', 'test'),
    'cifar100': ('train', 'test')
}

DATASET_NUM_CLASSES = {
    'imagenet2012': 1000,
    'cifar10': 10,
    'cifar100': 100
}


class DatasetBuilder(BaseDatasetBuilder):
    def __init__(self, config, image_size, num_devices, one_hot, is_train):
        super().__init__(config, is_train, num_devices)

        # Dataset params
        self._dataset_name = config.get('dataset', 'imagenet2012')
        self._as_supervised = True

        # Preprocessing params
        self._dtype = config.get('dtype', 'float32')
        self._num_preprocess_workers = config.get('workers', tf.data.experimental.AUTOTUNE)

        train_split, test_split = DATASET_SPLITS.get(self._dataset_name, ('train', 'validation'))
        self._split = train_split if is_train else test_split
        self._image_size = image_size
        self._num_classes = self._config.get('num_classes',
                                             DATASET_NUM_CLASSES.get(self._dataset_name))
        self._one_hot = one_hot
        self._cache = False
        self._shuffle_train = config.get('seed') is None
        self._shuffle_buffer_size = 10000
        self._deterministic_train = False
        self._use_slack = True

        self._preprocessing_fn = get_preprocessing(self._dataset_name,
                                                   self._config.model,
                                                   self._config.get('dataset_preprocessing_preset'))
        self._label_preprocessing_fn = \
            get_label_preprocessing_fn(self._dataset_name,
                                       self._num_classes,
                                       DATASET_NUM_CLASSES.get(self._dataset_name))

        self._tfrecord_datasets = records_dataset.__dict__

    @property
    def dtype(self):
        dtype_map = {
            'float32': tf.float32,
            'bfloat16': tf.bfloat16,
            'float16': tf.float16,
            'fp32': tf.float32,
            'bf16': tf.bfloat16,
        }
        dtype = dtype_map.get(self._dtype, None)
        if dtype is None:
            raise ValueError('Invalid DType provided. Supported types: {}'.format(dtype_map.keys()))

        return dtype

    @property
    def num_examples(self):
        if self._dataset_type == 'tfds':
            return self._dataset_loader.info.splits[self._split].num_examples
        if self._dataset_type == 'tfrecords':
            return self._dataset_loader.num_examples
        return None

    @property
    def num_classes(self):
        return self._num_classes

    def _get_loader_num_classes(self):
        if self._dataset_type == 'tfds':
            return self._dataset_loader.info.features['label'].num_classes
        if self._dataset_type == 'tfrecords':
            return self._dataset_loader.num_classes
        return None

    def _pipeline(self, dataset):
        if self.is_train and not self._cache:
            dataset = dataset.repeat()

        if self._dataset_type == 'tfrecords':
            dataset = dataset.prefetch(self.global_batch_size)

        if self._cache:
            dataset = dataset.cache()

        if self.is_train:
            if self._shuffle_train:
                dataset = dataset.shuffle(self._shuffle_buffer_size)
            dataset = dataset.repeat()

        if self._dataset_type == 'tfrecords':
            preprocess = lambda record: self._preprocess(*(self._dataset_loader.decoder(record)))
        else:
            preprocess = self._preprocess

        dataset = dataset.map(preprocess,
                              num_parallel_calls=self._num_preprocess_workers)

        dataset = dataset.batch(self.global_batch_size, drop_remainder=self.is_train)

        if self.is_train and self._deterministic_train is not None:
            options = tf.data.Options()
            options.experimental_deterministic = self._deterministic_train
            options.experimental_slack = self._use_slack
            options.experimental_optimization.parallel_batch = True
            options.experimental_optimization.map_fusion = True
            options.experimental_optimization.map_vectorization.enabled = True
            options.experimental_optimization.map_parallelization = True
            dataset = dataset.with_options(options)

        dataset = dataset.prefetch(self._num_devices)

        return dataset

    def _preprocess(self, image: tf.Tensor, label: tf.Tensor):
        image = self._preprocessing_fn(
            image,
            image_size=self._image_size,
            is_training=self.is_train,
            dtype=self.dtype)

        label = self._label_preprocessing_fn(label)
        label = tf.cast(label, tf.int32)
        if self._one_hot:
            label = tf.one_hot(label, self.num_classes)  # pylint: disable=E1120
            label = tf.reshape(label, [self.num_classes])

        return image, label
