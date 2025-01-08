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

from typing import Tuple

import tensorflow as tf

from examples.tensorflow.classification.datasets.preprocessing import utils

# Calculated from the CIFAR10 training set
CIFAR10_MEAN_RGB = (0.4914 * 255, 0.4822 * 255, 0.4465 * 255)
CIFAR10_STDDEV_RGB = (0.247 * 255, 0.2435 * 255, 0.2616 * 255)

# Calculated from the CIFAR100 training set
CIFAR100_MEAN_RGB = (0.5071 * 255, 0.4867 * 255, 0.4408 * 255)
CIFAR100_STDDEV_RGB = (0.2675 * 255, 0.2565 * 255, 0.2761 * 255)

IMAGE_SIZE = 32
PADDING = 4


def preprocess_for_eval(
    image: tf.Tensor,
    image_size: int = IMAGE_SIZE,
    dtype: tf.dtypes.DType = tf.float32,
    means: Tuple[float, ...] = None,
    stddev: Tuple[float, ...] = None,
) -> tf.Tensor:
    """
    Preprocesses the given image for evaluation.

    :param image: `Tensor` representing an image of arbitrary size.
    :param image_size: image height/width dimension.
    :param dtype: the dtype to convert the images to.
    :param means: values to subtract from each channel.
    :param stddev: values to divide from each channel.
    :return: a preprocessed and normalized image `Tensor`.
    """
    images = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
    images = tf.cast(images, tf.float32)
    images = utils.normalize(images, means, stddev)
    images = tf.image.convert_image_dtype(images, dtype=dtype)
    return images


def preprocess_for_train(
    image: tf.Tensor,
    image_size: int = IMAGE_SIZE,
    num_channels: int = 3,
    padding: int = PADDING,
    dtype: tf.dtypes.DType = tf.float32,
    means: Tuple[float, ...] = None,
    stddev: Tuple[float, ...] = None,
) -> tf.Tensor:
    """
    Preprocesses the given image for training.

    :param image: `Tensor` representing an image of arbitrary size.
    :param image_size: image height/width dimension.
    :param num_channels: number of image input channels.
    :param padding: the amound of padding before and after each dimension of the image.
    :param dtype: the dtype to convert the images to. Set to `None` to skip conversion.
    :param means: values to subtract from each channel.
    :param stddev: values to divide from each channel.
    :return: a preprocessed and normalized image `Tensor`.
    """
    images = image
    if padding > 0:
        images = tf.pad(images, [[padding, padding], [padding, padding], [0, 0]], constant_values=0)
    images = tf.image.random_crop(images, [image_size, image_size, num_channels])
    images = tf.image.random_flip_left_right(images)
    images = tf.cast(images, tf.float32)
    images = utils.normalize(images, means, stddev)
    images = tf.image.convert_image_dtype(images, dtype=dtype)
    return images


def preprocess_image(
    image: tf.Tensor,
    image_size: int = IMAGE_SIZE,
    is_training: bool = False,
    dtype: tf.dtypes.DType = tf.float32,
    means: Tuple[float, ...] = None,
    stddev: Tuple[float, ...] = None,
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
    :return: a preprocessed image.
    """
    if is_training:
        return preprocess_for_train(image=image, image_size=image_size, dtype=dtype, means=means, stddev=stddev)
    return preprocess_for_eval(image=image, image_size=image_size, dtype=dtype, means=means, stddev=stddev)


def cifar10_preprocess_image(
    image: tf.Tensor,
    image_size: int = IMAGE_SIZE,
    is_training: bool = False,
    dtype: tf.dtypes.DType = tf.float32,
    means: Tuple[float, ...] = CIFAR10_MEAN_RGB,
    stddev: Tuple[float, ...] = CIFAR10_STDDEV_RGB,
) -> tf.Tensor:
    """
    Preprocesses the given image using mean and standard deviation calculated by CIFAR10 dataset

    :param image: `Tensor` representing an image of arbitrary size.
    :param image_size: image height/width dimension
    :param is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
    :param dtype: the dtype to convert the images to.
    :param means: values to subtract from each channel.
    :param stddev: values to divide from each channel.
    :return: a preprocessed image.
    """
    return preprocess_image(
        image=image, image_size=image_size, is_training=is_training, dtype=dtype, means=means, stddev=stddev
    )


def cifar100_preprocess_image(
    image: tf.Tensor,
    image_size: int = IMAGE_SIZE,
    is_training: bool = False,
    dtype: tf.dtypes.DType = tf.float32,
    means: Tuple[float, ...] = CIFAR100_MEAN_RGB,
    stddev: Tuple[float, ...] = CIFAR100_STDDEV_RGB,
) -> tf.Tensor:
    """
    Preprocesses the given image using mean and standard deviation calculated by CIFAR100 dataset

    :param image: `Tensor` representing an image of arbitrary size.
    :param image_size: image height/width dimension
    :param is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
    :param dtype: the dtype to convert the images to.
    :param means: values to subtract from each channel.
    :param stddev: values to divide from each channel.
    :return: a preprocessed image.
    """
    return preprocess_image(
        image=image, image_size=image_size, is_training=is_training, dtype=dtype, means=means, stddev=stddev
    )
