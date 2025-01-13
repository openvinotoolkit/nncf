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

import nncf


def resize_image(image: tf.Tensor, height: int, width: int) -> tf.Tensor:
    """
    Resizes an image to a given height and width.

    :param image: `Tensor` representing an image of arbitrary size.
    :param height: image height.
    :param width: image width.
    :return: a float32 tensor containing the resized image.
    """
    return tf.compat.v1.image.resize(image, [height, width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)


def mean_image_subtraction(
    image: tf.Tensor, means: Tuple[float, ...], num_channels: int = 3, dtype: tf.dtypes.DType = tf.float32
) -> tf.Tensor:
    """
    Subtracts the given means from each image channel.

    :param image: a tensor of size [height, width, C].
    :param means: a C-vector of values to subtract from each channel.
    :param num_channels: number of color channels in the image that will be distorted.
    :param dtype: the dtype to convert the images to. Set to `None` to skip conversion.
    :return: the centered image.
    """
    if image.get_shape().ndims != 3:
        raise nncf.ValidationError("Input must be of size [height, width, C>0]")

    if len(means) != num_channels:
        raise nncf.ValidationError("len(means) must match the number of channels")

    means = tf.broadcast_to(means, tf.shape(image))
    if dtype is not None:
        means = tf.cast(means, dtype)

    return image - means


def standardize_image(
    image: tf.Tensor, stddev: Tuple[float, ...], num_channels: int = 3, dtype: tf.dtypes.DType = tf.float32
) -> tf.Tensor:
    """
    Divides the given stddev from each image channel.

    :param image: a tensor of size [height, width, C].
    :param stddev: a C-vector of values to divide from each channel.
    :param num_channels: number of color channels in the image that will be distorted.
    :param dtype: the dtype to convert the images to. Set to `None` to skip conversion.
    :return: the centered image.
    """
    if image.get_shape().ndims != 3:
        raise nncf.ValidationError("Input must be of size [height, width, C>0]")

    if len(stddev) != num_channels:
        raise nncf.ValidationError("len(stddev) must match the number of channels")

    stddev = tf.broadcast_to(stddev, tf.shape(image))
    if dtype is not None:
        stddev = tf.cast(stddev, dtype)

    return image / stddev


def normalize(
    image: tf.Tensor,
    means: Tuple[float, ...],
    stddev: Tuple[float, ...],
    num_channels: int = 3,
    dtype: tf.dtypes.DType = tf.float32,
) -> tf.Tensor:
    """
    Normalize a tensor image with mean and standard deviation.

    :param image: a tensor of size [height, width, C].
    :param means: a C-vector of values to subtract from each channel.
    :param stddev: a C-vector of values to divide from each channel.
    :param num_channels:  number of color channels in the image that will be distorted.
    :param dtype: the dtype to convert the images to. Set to `None` to skip conversion.
    :return: the normalized image.
    """
    dst = image
    if means:
        dst = mean_image_subtraction(dst, means, num_channels, dtype)
    if stddev:
        dst = standardize_image(dst, stddev, num_channels, dtype)
    return dst
