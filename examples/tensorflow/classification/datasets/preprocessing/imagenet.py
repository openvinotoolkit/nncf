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

import sys
from functools import partial
from typing import Tuple

import tensorflow as tf

from examples.tensorflow.classification.datasets.preprocessing import utils

# Calculated from the ImageNet training set
MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)

IMAGE_SIZE = 224
CROP_PADDING = 32


def get_model_normalize_fn(model_name: str, means: Tuple[float, ...] = None, stddev: Tuple[float, ...] = None):
    """
    Returns a function that normalizes the input image for a specific model.

    :param model_name: model name
    :param means: values to subtract from each channel.
    :param stddev: values to divide from each channel.
    :return: a function that normalizes the input image
    """
    if means or stddev:
        return partial(utils.normalize, means=means, stddev=stddev)

    if model_name in tf.keras.applications.__dict__:
        model_fn = tf.keras.applications.__dict__[model_name]
        return sys.modules[model_fn.__module__].preprocess_input

    return partial(utils.normalize, means=MEAN_RGB, stddev=STDDEV_RGB)


def center_crop(image: tf.Tensor, image_size: int = IMAGE_SIZE, crop_padding: int = CROP_PADDING) -> tf.Tensor:
    """
    Crops to center of image with padding then scales image_size.

    :param image: `Tensor` representing an image of arbitrary size.
    :param image_size: image height/width dimension.
    :param crop_padding: the padding size to use when centering the crop.
    :return: a cropped image `Tensor`.
    """
    shape = tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        ((image_size / (image_size + crop_padding)) * tf.cast(tf.minimum(image_height, image_width), tf.float32)),
        tf.int32,
    )

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2

    image = tf.image.crop_to_bounding_box(
        image,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=padded_center_crop_size,
        target_width=padded_center_crop_size,
    )

    image = utils.resize_image(image=image, height=image_size, width=image_size)

    return image


def crop_and_flip(image: tf.Tensor) -> tf.Tensor:
    """
    Crops an image to a random part of the image, then randomly flips.

    :param image: `Tensor` representing an image of arbitrary size.
    :return: a cropped image `Tensor`.
    """
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    shape = tf.shape(image)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True,
    )
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Reassemble the bounding box in the format the crop op requires.
    offset_height, offset_width, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    cropped = tf.image.crop_to_bounding_box(
        image,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=target_height,
        target_width=target_width,
    )

    # Flip to add a little more random distortion in.
    cropped = tf.image.random_flip_left_right(cropped)
    return cropped


def preprocess_for_eval(
    image: tf.Tensor, image_size: int = IMAGE_SIZE, dtype: tf.dtypes.DType = tf.float32, normalize_fn=None
) -> tf.Tensor:
    """
    Preprocesses the given image for evaluation.

    :param image: `Tensor` representing an image of arbitrary size.
    :param image_size: image height/width dimension.
    :param dtype: the dtype to convert the images to.
    :param normalize_fn: function which normalizes a tensor image with mean and standard deviation.
    :return: a preprocessed and normalized image `Tensor`.
    """
    images = center_crop(image, image_size)
    if normalize_fn:
        images = normalize_fn(images)
    images = tf.image.convert_image_dtype(images, dtype)
    return images


def preprocess_for_train(
    image: tf.Tensor, image_size: int = IMAGE_SIZE, dtype: tf.dtypes.DType = tf.float32, normalize_fn=None
) -> tf.Tensor:
    """
    Preprocesses the given image for training.

    :param image: `Tensor` representing an image of arbitrary size.
    :param image_size: image height/width dimension.
    :param dtype: the dtype to convert the images to.
    :param normalize_fn: function which normalizes a tensor image with mean and standard deviation.
    :return: a preprocessed and normalized image `Tensor`.
    """
    images = crop_and_flip(image=image)
    images = utils.resize_image(images, height=image_size, width=image_size)
    if normalize_fn:
        images = normalize_fn(images)
    images = tf.image.convert_image_dtype(images, dtype)
    return images


def imagenet_preprocess_image(
    image: tf.Tensor,
    image_size: int = IMAGE_SIZE,
    is_training: bool = False,
    dtype: tf.dtypes.DType = tf.float32,
    means: Tuple[float, ...] = None,
    stddev: Tuple[float, ...] = None,
    model_name: str = None,
) -> tf.Tensor:
    """
    Preprocesses the given image.

    :param image: `Tensor` representing an image of arbitrary size.
    :param image_size: image height/width dimension
    :param is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
    :param dtype: the dtype to convert the images to.
    :param means: values to subtract from each channel.
    :param stddev: values to divide from each channel.
    :param model_name: model name.
    :return: a preprocessed image.
    """
    normalize_fn = get_model_normalize_fn(model_name, means, stddev)

    if is_training:
        return preprocess_for_train(image=image, image_size=image_size, dtype=dtype, normalize_fn=normalize_fn)
    return preprocess_for_eval(image=image, image_size=image_size, dtype=dtype, normalize_fn=normalize_fn)


def imagenet_slim_preprocess_image(
    image: tf.Tensor, image_size: int = IMAGE_SIZE, is_training: bool = False, dtype: tf.dtypes.DType = tf.float32, **_
) -> tf.Tensor:
    image = tf.image.central_crop(image, central_fraction=0.875)
    if is_training:
        image = tf.image.random_flip_left_right(image)

    image = utils.resize_image(image, image_size, image_size)
    image.set_shape([image_size, image_size, 3])
    image = image / 255

    return tf.image.convert_image_dtype(image, dtype=dtype)


def imagenet_1000_to_1001_classes(label, class_diff=1, **_):
    return label + class_diff
