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

import os

import tensorflow as tf

from examples.tensorflow.common.tfrecords_dataset import TFRecordDataset

__all__ = ["imagenet2012"]

# ImageNet2012 specifications
NUM_TRAIN_EXAMPLES = 1281167
NUM_EVAL_EXAMPLES = 50000
NUM_CLASSES = 1000


def imagenet2012(config, is_train):
    return ImageNet2012(config, is_train)


def parse_record(record: tf.Tensor, has_background=False):
    """Parse an ImageNet record from a serialized string Tensor."""
    keys_to_features = {
        "image/encoded": tf.io.FixedLenFeature((), tf.string, ""),
        "image/format": tf.io.FixedLenFeature((), tf.string, "jpeg"),
        "image/class/label": tf.io.FixedLenFeature([], tf.int64, -1),
        "image/class/text": tf.io.FixedLenFeature([], tf.string, ""),
        "image/object/bbox/xmin": tf.io.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(dtype=tf.float32),
        "image/object/class/label": tf.io.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.io.parse_single_example(record, keys_to_features)

    label = tf.reshape(parsed["image/class/label"], shape=[1])
    label = tf.cast(label, tf.int32)

    # Subtract one so that labels are in [0, 1000)
    if not has_background:
        label -= 1

    encoded_image = tf.reshape(parsed["image/encoded"], shape=[])
    image = tf.image.decode_jpeg(encoded_image, channels=3)

    return image, label


class ImageNet2012(TFRecordDataset):
    def __init__(self, config, is_train):
        super().__init__(config, is_train)

        self._file_pattern = os.path.join(self.dataset_dir, "{}-*-of-*".format("train" if is_train else "validation"))

    @property
    def num_examples(self):
        if self.is_train:
            return NUM_TRAIN_EXAMPLES
        return NUM_EVAL_EXAMPLES

    @property
    def num_classes(self):
        return NUM_CLASSES

    @property
    def file_pattern(self):
        return self._file_pattern

    @property
    def decoder(self):
        return parse_record
