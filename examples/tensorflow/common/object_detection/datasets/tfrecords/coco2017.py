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

__all__ = ["coco2017"]

# https://www.tensorflow.org/datasets/catalog/coco#coco2017
NUM_EXAMPLES_TRAIN = 118287
NUM_EXAMPLES_EVAL = 5000
NUM_CLASSES = 80


def coco2017(config, is_train):
    return COCO2017(config, is_train)


def parse_record(record, include_mask=False, model=None, is_train=None):
    """Parse a COCO2017 record from a serialized string Tensor.

    Args:
        recird: a single serialized tf.Example string.

    Returns:
        decoded_tensors: a dictionary of tensors with the following fields:
            - image: a uint8 tensor of shape [None, None, 3].
            - source_id: a string scalar tensor.
            - height: an integer scalar tensor.
            - width: an integer scalar tensor.
            - groundtruth_classes: a int64 tensor of shape [None].
            - groundtruth_is_crowd: a bool tensor of shape [None].
            - groundtruth_area: a float32 tensor of shape [None].
            - groundtruth_boxes: a float32 tensor of shape [None, 4].
            - groundtruth_instance_masks: a float32 tensor of shape [None, None, None].
            - groundtruth_instance_masks_png: a string tensor of shape [None].
    """

    def _decode_image(parsed_tensors, model):
        """Decodes the image and set its static shape."""
        if model == "YOLOv4":
            image = tf.image.decode_jpeg(parsed_tensors["image/encoded"], channels=3, dct_method="INTEGER_ACCURATE")
        else:
            image = tf.io.decode_image(parsed_tensors["image/encoded"], channels=3)
        image.set_shape([None, None, 3])
        return image

    def _decode_boxes(parsed_tensors):
        """Concat box coordinates in the format of [ymin, xmin, ymax, xmax]."""
        xmin = parsed_tensors["image/object/bbox/xmin"]
        xmax = parsed_tensors["image/object/bbox/xmax"]
        ymin = parsed_tensors["image/object/bbox/ymin"]
        ymax = parsed_tensors["image/object/bbox/ymax"]
        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    def _decode_areas(parsed_tensors):
        xmin = parsed_tensors["image/object/bbox/xmin"]
        xmax = parsed_tensors["image/object/bbox/xmax"]
        ymin = parsed_tensors["image/object/bbox/ymin"]
        ymax = parsed_tensors["image/object/bbox/ymax"]
        return tf.cond(
            tf.greater(tf.shape(parsed_tensors["image/object/area"])[0], 0),
            lambda: parsed_tensors["image/object/area"],
            lambda: (xmax - xmin) * (ymax - ymin),
        )

    def _decode_masks(parsed_tensors):
        """Decode a set of PNG masks to the tf.float32 tensors."""

        def _decode_png_mask(png_bytes):
            mask = tf.squeeze(tf.io.decode_png(png_bytes, channels=1, dtype=tf.uint8), axis=-1)
            mask = tf.cast(mask, tf.float32)
            mask.set_shape([None, None])
            return mask

        height = parsed_tensors["image/height"]
        width = parsed_tensors["image/width"]
        masks = parsed_tensors["image/object/mask"]
        return tf.cond(
            tf.greater(tf.size(input=masks), 0),
            lambda: tf.map_fn(_decode_png_mask, masks, dtype=tf.float32),
            lambda: tf.zeros([0, height, width], dtype=tf.float32),
        )

    keys_to_features = {
        "image/encoded": tf.io.FixedLenFeature((), tf.string),
        "image/source_id": tf.io.FixedLenFeature((), tf.string),
        "image/height": tf.io.FixedLenFeature((), tf.int64),
        "image/width": tf.io.FixedLenFeature((), tf.int64),
        "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
        "image/object/class/label": tf.io.VarLenFeature(tf.int64),
        "image/object/area": tf.io.VarLenFeature(tf.float32),
        "image/object/is_crowd": tf.io.VarLenFeature(tf.int64),
    }

    if include_mask:
        keys_to_features.update(
            {
                "image/object/mask": tf.io.VarLenFeature(tf.string),
            }
        )

    parsed_tensors = tf.io.parse_single_example(serialized=record, features=keys_to_features)
    for k in parsed_tensors:
        if isinstance(parsed_tensors[k], tf.SparseTensor):
            if parsed_tensors[k].dtype == tf.string:
                parsed_tensors[k] = tf.sparse.to_dense(parsed_tensors[k], default_value="")
            else:
                parsed_tensors[k] = tf.sparse.to_dense(parsed_tensors[k], default_value=0)

    image = _decode_image(parsed_tensors, model)
    boxes = _decode_boxes(parsed_tensors)
    areas = _decode_areas(parsed_tensors)

    if include_mask:
        masks = _decode_masks(parsed_tensors)

    is_crowds = tf.cond(
        tf.greater(tf.shape(parsed_tensors["image/object/is_crowd"])[0], 0),
        lambda: tf.cast(parsed_tensors["image/object/is_crowd"], tf.bool),
        lambda: tf.zeros_like(parsed_tensors["image/object/class/label"], dtype=tf.bool),
    )

    def _convert_labels_to_80_classes(parsed_tensors):
        # 1..90 --> 0..79
        match = tf.constant(
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                27,
                28,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                67,
                70,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                84,
                85,
                86,
                87,
                88,
                89,
                90,
            ],
            dtype=tf.int64,
        )

        labels = parsed_tensors["image/object/class/label"]
        labels = tf.reshape(labels, (tf.shape(labels)[0], 1))
        labels = tf.where(tf.equal(match, labels))[:, -1]
        return labels

    labels = parsed_tensors["image/object/class/label"]
    if model == "YOLOv4" and is_train:
        labels = _convert_labels_to_80_classes(parsed_tensors)

    decoded_tensors = {
        "image": image,
        "source_id": parsed_tensors["image/source_id"],
        "height": parsed_tensors["image/height"],
        "width": parsed_tensors["image/width"],
        "groundtruth_classes": labels,
        "groundtruth_is_crowd": is_crowds,
        "groundtruth_area": areas,
        "groundtruth_boxes": boxes,
    }

    if include_mask:
        decoded_tensors.update(
            {
                "groundtruth_instance_masks": masks,
                "groundtruth_instance_masks_png": parsed_tensors["image/object/mask"],
            }
        )

    return decoded_tensors


class COCO2017(TFRecordDataset):
    def __init__(self, config, is_train):
        super().__init__(config, is_train)

        self._file_pattern = os.path.join(self.dataset_dir, "{}-*-of-*".format("train" if is_train else "val"))

        self._val_json_file = os.path.join(self.dataset_dir, "instances_val2017.json")

        config.val_json_file = self._val_json_file

    @property
    def num_examples(self):
        return NUM_EXAMPLES_TRAIN if self.is_train else NUM_EXAMPLES_EVAL

    @property
    def num_classes(self):
        return NUM_CLASSES

    @property
    def file_pattern(self):
        return self._file_pattern

    @property
    def decoder(self):
        return parse_record
