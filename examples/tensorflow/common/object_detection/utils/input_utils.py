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

import math

import tensorflow as tf

from examples.tensorflow.common.object_detection.utils import box_utils


def pad_to_fixed_size(input_tensor, size, constant_values=0):
    """
    Pads data to a fixed length at the first dimension.

    :param input_tensor: `Tensor` with any dimension.
    :param size: `int` number for the first dimension of output Tensor.
      constant_values: `int` value assigned to the paddings.
    :return: `Tensor` with the first dimension padded to `size`.
    """
    input_shape = input_tensor.get_shape().as_list()
    padding_shape = []

    # Computes the padding length on the first dimension.
    padding_length = tf.maximum(0, size - tf.shape(input_tensor)[0])
    assert_length = tf.Assert(tf.greater_equal(padding_length, 0), [padding_length])
    with tf.control_dependencies([assert_length]):
        padding_shape.append(padding_length)

    # Copies shapes of the rest of input shape dimensions.
    for i in range(1, len(input_shape)):
        padding_shape.append(tf.shape(input=input_tensor)[i])

    # Pads input tensor to the fixed first dimension.
    paddings = tf.cast(constant_values * tf.ones(padding_shape), input_tensor.dtype)
    padded_tensor = tf.concat([input_tensor, paddings], 0)
    output_shape = input_shape
    output_shape[0] = size
    padded_tensor.set_shape(output_shape)

    return padded_tensor


def normalize_image(image, offset=(0.485, 0.456, 0.406), scale=(0.229, 0.224, 0.225)):
    """Normalizes the image to zero mean and unit variance."""
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    offset = tf.constant(offset)
    offset = tf.expand_dims(offset, axis=0)
    offset = tf.expand_dims(offset, axis=0)
    image -= offset

    scale = tf.constant(scale)
    scale = tf.expand_dims(scale, axis=0)
    scale = tf.expand_dims(scale, axis=0)
    image /= scale
    return image


def compute_padded_size(desired_size, stride):
    """
    Compute the padded size given the desired size and the stride.

    The padded size will be the smallest rectangle, such that each dimension is
    the smallest multiple of the stride which is larger than the desired
    dimension. For example, if desired_size = (100, 200) and stride = 32,
    the output padded_size = (128, 224).

    :param desired_size: a `Tensor` or `int` list/tuple of two elements representing
        [height, width] of the target output image size.
    :param stride: an integer, the stride of the backbone network.
    :return padded_size: a `Tensor` or `int` list/tuple of two elements representing
        [height, width] of the padded output image size.
    """
    if isinstance(desired_size, (list, tuple)):
        padded_size = [int(math.ceil(d * 1.0 / stride) * stride) for d in desired_size]
    else:
        padded_size = tf.cast(tf.math.ceil(tf.cast(desired_size, tf.float32) / stride) * stride, tf.int32)
    return padded_size


def resize_and_crop_image(
    image,
    desired_size,
    padded_size,
    aug_scale_min=1.0,
    aug_scale_max=1.0,
    seed=1,
    method=tf.image.ResizeMethod.BILINEAR,
):
    """
    Resizes the input image to output size.

    Resize and pad images given the desired output size of the image and
    stride size.

    Here are the preprocessing steps.
    1. For a given image, keep its aspect ratio and rescale the image to make it
      the largest rectangle to be bounded by the rectangle specified by the
      `desired_size`.
    2. Pad the rescaled image to the padded_size.

    :param image: a `Tensor` of shape [height, width, 3] representing an image.
    :param desired_size: a `Tensor` or `int` list/tuple of two elements representing
        [height, width] of the desired actual output image size.
    :param padded_size: a `Tensor` or `int` list/tuple of two elements representing
        [height, width] of the padded output image size. Padding will be applied
        after scaling the image to the desired_size.
    :param aug_scale_min: a `float` with range between [0, 1.0] representing minimum
        random scale applied to desired_size for training scale jittering.
    :param aug_scale_max: a `float` with range between [1.0, inf] representing maximum
        random scale applied to desired_size for training scale jittering.
    :param seed: seed for random scale jittering.
    :param method: function to resize input image to scaled image.
    :return output_image: `Tensor` of shape [height, width, 3] where [height, width]
        equals to `output_size`.
    :return image_info: a 2D `Tensor` that encodes the information of the image and the
        applied preprocessing. It is in the format of
        [[original_height, original_width], [desired_height, desired_width],
        [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
        desireed_width] is the actual scaled image size, and [y_scale, x_scale] is
        the scaling factory, which is the ratio of
        scaled dimension / original dimension.
    """
    with tf.name_scope("resize_and_crop_image"):
        image_size = tf.cast(tf.shape(input=image)[0:2], tf.float32)
        random_jittering = aug_scale_min != 1.0 or aug_scale_max != 1.0

        if random_jittering:
            random_scale = tf.random.uniform([], aug_scale_min, aug_scale_max, tf.float32, seed)
            scaled_size = tf.round(random_scale * desired_size)
        else:
            scaled_size = desired_size

        scale = tf.minimum(scaled_size[0] / image_size[0], scaled_size[1] / image_size[1])
        scaled_size = tf.round(image_size * scale)

        # Computes 2D image_scale.
        image_scale = scaled_size / image_size

        # Selects non-zero random offset (x, y) if scaled image is larger than
        # desired_size.
        if random_jittering:
            max_offset = scaled_size - desired_size
            max_offset = tf.where(tf.less(max_offset, 0), tf.zeros_like(max_offset), max_offset)
            offset = max_offset * tf.random.uniform(
                [
                    2,
                ],
                0,
                1,
                tf.float32,
                seed,
            )
            offset = tf.cast(offset, tf.int32)
        else:
            offset = tf.zeros((2,), tf.int32)

        scaled_image = tf.image.resize(image, tf.cast(scaled_size, tf.int32), method=method)

        if random_jittering:
            scaled_image = scaled_image[
                offset[0] : offset[0] + desired_size[0], offset[1] : offset[1] + desired_size[1], :
            ]

        output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0, padded_size[0], padded_size[1])

        image_info = tf.stack([image_size, tf.cast(desired_size, tf.float32), image_scale, tf.cast(offset, tf.float32)])

        return output_image, image_info


def resize_and_crop_image_v2(
    image,
    short_side,
    long_side,
    padded_size,
    aug_scale_min=1.0,
    aug_scale_max=1.0,
    seed=1,
    method=tf.image.ResizeMethod.BILINEAR,
):
    """
    Resizes the input image to output size (Faster R-CNN style).

    Resize and pad images given the specified short / long side length and the
    stride size.

    Here are the preprocessing steps.
    1. For a given image, keep its aspect ratio and first try to rescale the short
      side of the original image to `short_side`.
    2. If the scaled image after 1 has a long side that exceeds `long_side`, keep
      the aspect ratio and rescal the long side of the image to `long_side`.
    2. Pad the rescaled image to the padded_size.

    :param image: a `Tensor` of shape [height, width, 3] representing an image.
    :param short_side: a scalar `Tensor` or `int` representing the desired short side
        to be rescaled to.
    :param long_side: a scalar `Tensor` or `int` representing the desired long side to
        be rescaled to.
    :param padded_size: a `Tensor` or `int` list/tuple of two elements representing
        [height, width] of the padded output image size. Padding will be applied
        after scaling the image to the desired_size.
    :param aug_scale_min: a `float` with range between [0, 1.0] representing minimum
        random scale applied to desired_size for training scale jittering.
    :param aug_scale_max: a `float` with range between [1.0, inf] representing maximum
        random scale applied to desired_size for training scale jittering.
    :param seed: seed for random scale jittering.
      method: function to resize input image to scaled image.
    :return output_image: `Tensor` of shape [height, width, 3] where [height, width]
        equals to `output_size`.
    :return image_info: a 2D `Tensor` that encodes the information of the image and the
        applied preprocessing. It is in the format of
        [[original_height, original_width], [desired_height, desired_width],
        [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
        desired_width] is the actual scaled image size, and [y_scale, x_scale] is
        the scaling factor, which is the ratio of
        scaled dimension / original dimension.
    """
    with tf.name_scope("resize_and_crop_image_v2"):
        image_size = tf.cast(tf.shape(image)[0:2], tf.float32)

        scale_using_short_side = short_side / tf.math.minimum(image_size[0], image_size[1])
        scale_using_long_side = long_side / tf.math.maximum(image_size[0], image_size[1])

        scaled_size = tf.math.round(image_size * scale_using_short_side)
        scaled_size = tf.where(
            tf.math.greater(tf.math.maximum(scaled_size[0], scaled_size[1]), long_side),
            tf.math.round(image_size * scale_using_long_side),
            scaled_size,
        )
        desired_size = scaled_size

        random_jittering = aug_scale_min != 1.0 or aug_scale_max != 1.0

        if random_jittering:
            random_scale = tf.random.uniform([], aug_scale_min, aug_scale_max, tf.float32, seed)
            scaled_size = tf.math.round(random_scale * scaled_size)

        # Computes 2D image_scale.
        image_scale = scaled_size / image_size

        # Selects non-zero random offset (x, y) if scaled image is larger than
        # desired_size.
        if random_jittering:
            max_offset = scaled_size - desired_size
            max_offset = tf.where(tf.math.less(max_offset, 0), tf.zeros_like(max_offset), max_offset)
            offset = max_offset * tf.random.uniform(
                [
                    2,
                ],
                0,
                1,
                tf.float32,
                seed,
            )
            offset = tf.cast(offset, tf.int32)
        else:
            offset = tf.zeros((2,), tf.int32)

        scaled_image = tf.image.resize(image, tf.cast(scaled_size, tf.int32), method=method)

        if random_jittering:
            scaled_image = scaled_image[
                offset[0] : offset[0] + desired_size[0], offset[1] : offset[1] + desired_size[1], :
            ]

        output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0, padded_size[0], padded_size[1])

        image_info = tf.stack([image_size, tf.cast(desired_size, tf.float32), image_scale, tf.cast(offset, tf.float32)])

        return output_image, image_info


def resize_and_crop_boxes(boxes, image_scale, output_size, offset):
    """
    Resizes boxes to output size with scale and offset.

    :param boxes: `Tensor` of shape [N, 4] representing ground truth boxes.
      image_scale: 2D float `Tensor` representing scale factors that apply to
        [height, width] of input image.
    :param output_size: 2D `Tensor` or `int` representing [height, width] of target
        output image size.
    :param offset: 2D `Tensor` representing top-left corner [y0, x0] to crop scaled
        boxes.
    :return boxes: `Tensor` of shape [N, 4] representing the scaled boxes.
    """
    # Adjusts box coordinates based on image_scale and offset.
    boxes *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
    boxes -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
    # Clips the boxes.
    boxes = box_utils.clip_boxes(boxes, output_size)

    return boxes


def random_horizontal_flip(image, boxes=None, masks=None, seed=None):
    """
    Randomly flips the image and detections horizontally.
    The probability of flipping the image is 50%.

    :param image: rank 3 float32 tensor with shape [height, width, channels].
    :param boxes: (optional) rank 2 float32 tensor with shape [N, 4] containing the
        bounding boxes. Boxes are in normalized form meaning their coordinates
        vary between [0, 1]. Each row is in the form of [ymin, xmin, ymax, xmax].
    :param masks: (optional) rank 3 float32 tensor with shape [num_instances, height,
        width] containing instance masks. The masks are of the same height, width
        as the input `image`.
    :param seed: random seed
    :return image: image which is the same shape as input image.
    :return boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
        Boxes are in normalized form meaning their coordinates vary
        between [0, 1].
    :return masks: rank 3 float32 tensor with shape [num_instances, height, width]
        containing instance masks.
    """

    def _flip_image(image):
        image_flipped = tf.image.flip_left_right(image)
        return image_flipped

    def _flip_boxes_left_right(boxes):
        """
        Left-right flip the boxes.

        :param boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4]. Boxes
            are in normalized form meaning their coordinates vary between [0, 1]. Each
            row is in the form of [ymin, xmin, ymax, xmax].
        :return: Flipped boxes.
        """
        ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=1)
        flipped_xmin = tf.subtract(1.0, xmax)
        flipped_xmax = tf.subtract(1.0, xmin)
        flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], 1)
        return flipped_boxes

    def _flip_masks_left_right(masks):
        """
        Left-right flip masks.

        :param masks: rank 3 float32 tensor with shape [num_instances, height, width]
                representing instance masks.
        :return flipped masks: rank 3 float32 tensor with shape
                [num_instances, height, width] representing instance masks.
        """
        return masks[:, :, ::-1]

    with tf.name_scope("RandomHorizontalFlip"):
        result = []
        # random variable defining whether to do flip or not
        do_a_flip_random = tf.greater(tf.random.uniform([], seed=seed, dtype=tf.float32), 0.5)

        # flip image
        image = tf.cond(pred=do_a_flip_random, true_fn=lambda: _flip_image(image), false_fn=lambda: image)
        result.append(image)

        # flip boxes
        if boxes is not None:
            boxes = tf.cond(
                pred=do_a_flip_random, true_fn=lambda: _flip_boxes_left_right(boxes), false_fn=lambda: boxes
            )
            result.append(boxes)

        # flip masks
        if masks is not None:
            masks = tf.cond(
                pred=do_a_flip_random, true_fn=lambda: _flip_masks_left_right(masks), false_fn=lambda: masks
            )
            result.append(masks)

        return tuple(result)


def random_vertical_flip(image, boxes=None, masks=None, probability=0.1, seed=None):
    """
    Randomly flips the image and detections vertically.

    :param image: rank 3 float32 tensor with shape [height, width, channels].
    :param boxes: (optional) rank 2 float32 tensor with shape [N, 4] containing the
        bounding boxes. Boxes are in normalized form meaning their coordinates
        vary between [0, 1]. Each row is in the form of [ymin, xmin, ymax, xmax].
    :param masks: (optional) rank 3 float32 tensor with shape [num_instances, height,
        width] containing instance masks. The masks are of the same height, width
        as the input `image`.
    :param seed: random seed
    :return image: image which is the same shape as input image.
    :return boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
        Boxes are in normalized form meaning their coordinates vary
        between [0, 1].
    :return masks: rank 3 float32 tensor with shape [num_instances, height, width]
        containing instance masks.
    """

    def _flip_image(image):
        image_flipped = tf.image.flip_up_down(image)
        return image_flipped

    def _flip_boxes_up_down(boxes):
        """Up-down flip the boxes.
        Args:
        :param boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
                 Boxes are in normalized form meaning their coordinates vary
                 between [0, 1].
                 Each row is in the form of [ymin, xmin, ymax, xmax].
        Returns:
          Flipped boxes.
        """
        ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=1)
        flipped_ymin = tf.subtract(1.0, ymax)
        flipped_ymax = tf.subtract(1.0, ymin)
        flipped_boxes = tf.concat([flipped_ymin, xmin, flipped_ymax, xmax], 1)
        return flipped_boxes

    def _flip_masks_up_down(masks):
        """
        Up-down flip masks.

        :param masks: rank 3 float32 tensor with shape
            [num_instances, height, width] representing instance masks.
        :return flipped masks: rank 3 float32 tensor with shape
            [num_instances, height, width] representing instance masks.
        """
        return masks[:, ::-1, :]

    with tf.name_scope("RandomVerticalFlip"):
        result = []
        # random variable defining whether to do flip or not
        do_a_flip_random = tf.greater(tf.random.uniform([], seed=seed, dtype=tf.float32), probability)

        # flip image
        image = tf.cond(pred=do_a_flip_random, true_fn=lambda: _flip_image(image), false_fn=lambda: image)
        result.append(image)

        # flip boxes
        if boxes is not None:
            boxes = tf.cond(pred=do_a_flip_random, true_fn=lambda: _flip_boxes_up_down(boxes), false_fn=lambda: boxes)
            result.append(boxes)

        # flip masks
        if masks is not None:
            masks = tf.cond(pred=do_a_flip_random, true_fn=lambda: _flip_masks_up_down(masks), false_fn=lambda: masks)
            result.append(masks)

        return tuple(result)


def random_rotation90(image, boxes=None, masks=None, probability=0.1, seed=None):
    """
    Randomly rotates the image and detections 90 degrees counter-clockwise.
    The probability of rotating the image is 50%. This can be combined with
    random_horizontal_flip and random_vertical_flip to produce an output with a
    uniform distribution of the eight possible 90 degree rotation / reflection
    combinations.

    :param image: rank 3 float32 tensor with shape [height, width, channels].
    :param boxes: (optional) rank 2 float32 tensor with shape [N, 4]
           containing the bounding boxes.
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    :param masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width] containing instance masks. The masks
           are of the same height, width as the input `image`.
    :param probability: the probability of performing this augmentation.
    :param seed: random seed
    :return image: image which is the same shape as input image.
    If boxes and masks, are not None,
    the function also returns the following tensors.
    :return boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
    :return masks: rank 3 float32 tensor with shape [num_instances, height, width]
           containing instance masks.
    """

    def _rot90_image(image):
        # flip image
        image_rotated = tf.image.rot90(image)
        return image_rotated

    def _rot90_boxes(boxes):
        """
        Rotate boxes counter-clockwise by 90 degrees.

        :param boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
               Boxes are in normalized form meaning their coordinates vary
               between [0, 1].
               Each row is in the form of [ymin, xmin, ymax, xmax].
        :return: Rotated boxes.
        """
        ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=1)
        rotated_ymin = tf.subtract(1.0, xmax)
        rotated_ymax = tf.subtract(1.0, xmin)
        rotated_xmin = ymin
        rotated_xmax = ymax
        rotated_boxes = tf.concat([rotated_ymin, rotated_xmin, rotated_ymax, rotated_xmax], 1)
        return rotated_boxes

    def _rot90_masks(masks):
        """
        Rotate masks counter-clockwise by 90 degrees.

        :param masks: rank 3 float32 tensor with shape
          [num_instances, height, width] representing instance masks.
        :return rotated masks: rank 3 float32 tensor with shape
          [num_instances, height, width] representing instance masks.
        """
        masks = tf.transpose(masks, [0, 2, 1])
        return masks[:, ::-1, :]

    with tf.name_scope("RandomRotation90"):
        result = []
        # random variable defining whether to do flip or not
        do_a_rot90_random = tf.greater(tf.random.uniform([], seed=seed, dtype=tf.float32), probability)

        # flip image
        image = tf.cond(do_a_rot90_random, lambda: _rot90_image(image), lambda: image)
        result.append(image)

        # flip boxes
        if boxes is not None:
            boxes = tf.cond(do_a_rot90_random, lambda: _rot90_boxes(boxes), lambda: boxes)
            result.append(boxes)

        # flip masks
        if masks is not None:
            masks = tf.cond(do_a_rot90_random, lambda: _rot90_masks(masks), lambda: masks)
            result.append(masks)

        return tuple(result)


def random_adjust_brightness(image, max_delta=0.2, seed=None):
    """
    Randomly adjusts brightness.
    Makes sure the output image is still between 0 and 255.

    :param image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    :param max_delta: how much to change the brightness. A value between [0, 1).
    :param seed: random seed.
    :return image: image which is the same shape as input image.
    """
    with tf.name_scope("RandomAdjustBrightness"):
        # random variable from [-max_delta, max_delta]
        delta = tf.random.uniform([], -max_delta, max_delta, tf.float32, seed)

        def _adjust_brightness(image):
            image = tf.image.adjust_brightness(image / 255, delta) * 255
            image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
            return image

        image = _adjust_brightness(image)

        return image


def random_adjust_saturation(image, min_delta=0.8, max_delta=1.25, seed=None):
    """
    Randomly adjusts saturation.
    Makes sure the output image is still between 0 and 255.

    :param image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    :param min_delta: see max_delta.
    :param max_delta: how much to change the saturation. Saturation will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current saturation of the image.
    :param seed: random seed.
    :return image: image which is the same shape as input image.
    """
    with tf.name_scope("RandomAdjustSaturation"):
        saturation_factor = tf.random.uniform([], min_delta, max_delta, tf.float32, seed)

        def _adjust_saturation(image):
            image = tf.image.adjust_saturation(image / 255, saturation_factor) * 255
            image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
            return image

        image = _adjust_saturation(image)
        return image


def random_adjust_contrast(image, min_delta=0.8, max_delta=1.25, seed=None):
    """
    Randomly adjusts contrast.
    Makes sure the output image is still between 0 and 255.

    :param image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    :param min_delta: see max_delta.
    :param max_delta: how much to change the contrast. Contrast will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current contrast of the image.
    :param seed: random seed.
    :return image: image which is the same shape as input image.
    """
    with tf.name_scope("RandomAdjustContrast"):
        contrast_factor = tf.random.uniform([], min_delta, max_delta, tf.float32, seed)

        def _adjust_contrast(image):
            image = tf.image.adjust_contrast(image / 255, contrast_factor) * 255
            image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
            return image

        image = _adjust_contrast(image)
        return image


def random_adjust_hue(image, max_delta=0.02, seed=None):
    """
    Randomly adjusts hue.
    Makes sure the output image is still between 0 and 255.

    :param image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    :param max_delta: change hue randomly with a value between 0 and max_delta.
    :param seed: random seed.
    :return image: image which is the same shape as input image.
    """
    with tf.name_scope("RandomAdjustHue"):
        delta = tf.random.uniform([], -max_delta, max_delta, tf.float32, seed)

        def _adjust_hue(image):
            image = tf.image.adjust_hue(image / 255, delta) * 255
            image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
            return image

        image = _adjust_hue(image)
        return image


def random_rgb_to_gray(image, probability=0.2, seed=None):
    """
    Changes the image from RGB to Grayscale with the given probability.

    :param image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    :param probability: the probability of returning a grayscale image.
            The probability should be a number between [0, 1].
    :param seed: random seed.
    :return image: image which is the same shape as input image.
    """

    def _image_to_gray(image):
        image_gray1 = tf.image.rgb_to_grayscale(image)
        image_gray3 = tf.image.grayscale_to_rgb(image_gray1)
        return image_gray3

    with tf.name_scope("RandomRGBtoGray"):
        do_gray_random = tf.greater(tf.random.uniform([], seed=seed, dtype=tf.float32), probability)

        image = tf.cond(do_gray_random, lambda: _image_to_gray(image), lambda: image)

    return image
