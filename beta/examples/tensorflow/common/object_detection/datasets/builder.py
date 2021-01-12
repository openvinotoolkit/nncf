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

from functools import partial
import tensorflow as tf
import examples.tensorflow.common.object_detection.datasets.tfrecords as records_dataset
from examples.tensorflow.common.dataset_builder import BaseDatasetBuilder
from examples.tensorflow.common.object_detection.datasets.preprocessing_selector import get_preprocess_input_fn


class COCODatasetBuilder(BaseDatasetBuilder):
    """COCO2017 dataset loader and input processing."""
    def __init__(self, config, is_train, num_devices):
        super().__init__(config, is_train, num_devices)

        # Pipeline params
        self._shuffle_buffer_size = 1000
        self._num_preprocess_workers = config.get('workers', tf.data.experimental.AUTOTUNE)
        self._cache = False
        self._include_mask = config.get('include_mask', False)

        # TFDS params
        self._skip_decoding = True

        self._tfrecord_datasets = records_dataset.__dict__

    @property
    def num_examples(self):
        if self._dataset_type == 'tfds':
            return self._dataset_loader.info.splits[self._split].num_examples
        if self._dataset_type == 'tfrecords':
            return self._dataset_loader.num_examples
        return None

    @property
    def num_classes(self):
        if self._dataset_type == 'tfds':
            return self._dataset_loader.info.features['objects']['label'].num_classes
        if self._dataset_type == 'tfrecords':
            return self._dataset_loader.num_classes
        return None

    def _tfds_decoder(self, features_dict):
        def _decode_image(features):
            image = tf.io.decode_image(features['image'], channels=3)
            image.set_shape([None, None, 3])
            return image

        def _convert_labels_to_91_classes(features):
            # 0..79 --> 0..90
            match = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
                                 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                                 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                                 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                                 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
                                 80, 81, 82, 84, 85, 86, 87, 88, 89, 90], dtype=tf.int64)

            labels = features['objects']['label']
            labels = tf.gather(match, labels, axis=None)
            return labels

        image = _decode_image(features_dict)
        labels = _convert_labels_to_91_classes(features_dict)

        decoded_tensors = {
            'image': image,
            'source_id': tf.cast(features_dict['image/id'], tf.int32),
            'groundtruth_classes': labels,
            'groundtruth_is_crowd': features_dict['objects']['is_crowd'],
            'groundtruth_area': features_dict['objects']['area'],
            'groundtruth_boxes': features_dict['objects']['bbox'],
        }

        return decoded_tensors

    def _pipeline(self, dataset):
        if self._cache:
            dataset = dataset.cache()

        if self._is_train:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(self._shuffle_buffer_size)

        if self._dataset_type == 'tfrecords':
            decoder_fn = partial(self._dataset_loader.decoder, include_mask=self._include_mask)
        else:
            decoder_fn = self._tfds_decoder

        preprocess_input_fn = get_preprocess_input_fn(self._config, self._is_train)
        preprocess_pipeline = lambda record: preprocess_input_fn(decoder_fn(record))

        dataset = dataset.map(preprocess_pipeline, num_parallel_calls=self._num_preprocess_workers)
        dataset = dataset.batch(self.global_batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
