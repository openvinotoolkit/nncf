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

def center_crop(image: tf.Tensor,
                image_size: int,
                crop_padding: int) -> tf.Tensor:
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
        ((image_size / (image_size + crop_padding)) *
         tf.cast(tf.minimum(image_height, image_width), tf.float32)),
        tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2

    image = tf.image.crop_to_bounding_box(
        image,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=padded_center_crop_size,
        target_width=padded_center_crop_size)

    image = tf.compat.v1.image.resize(
        image,
        [image_size, image_size],
        method=tf.image.ResizeMethod.BILINEAR,
        align_corners=False)

    return image
