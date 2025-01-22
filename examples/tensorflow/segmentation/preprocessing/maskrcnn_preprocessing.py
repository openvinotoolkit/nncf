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

import tensorflow as tf

from examples.tensorflow.common.object_detection.utils import anchor
from examples.tensorflow.common.object_detection.utils import box_utils
from examples.tensorflow.common.object_detection.utils import dataloader_utils
from examples.tensorflow.common.object_detection.utils import input_utils


class MaskRCNNPreprocessor:
    """Parser to parse an image and its annotations into a dictionary of tensors."""

    def __init__(self, config, is_train):
        """Initializes parameters for parsing annotations in the dataset.

        Attributes:
          output_size: `Tensor` or `list` for [height, width] of output image. The
              output_size should be divided by the largest feature stride 2^max_level.
          min_level: `int` number of minimum level of the output feature pyramid.
          max_level: `int` number of maximum level of the output feature pyramid.
          num_scales: `int` number representing intermediate scales added
              on each level. For instances, num_scales=2 adds one additional
              intermediate anchor scales [2^0, 2^0.5] on each level.
          aspect_ratios: `list` of float numbers representing the aspect raito
              anchors added on each level. The number indicates the ratio of width to
              height. For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors
              on each scale level.
          anchor_size: `float` number representing the scale of size of the base
              anchor to the feature stride 2^level.
          rpn_match_threshold:
          rpn_unmatched_threshold:
          rpn_batch_size_per_im:
          rpn_fg_fraction:
          aug_rand_hflip: `bool`, if True, augment training with random
              horizontal flip.
          aug_scale_min: `float`, the minimum scale applied to `output_size` for
              data augmentation during training.
          aug_scale_max: `float`, the maximum scale applied to `output_size` for
              data augmentation during training.
          skip_crowd_during_training: `bool`, if True, skip annotations labeled with
              `is_crowd` equals to 1.
          max_num_instances: `int` number of maximum number of instances in an
              image. The groundtruth data will be padded to `max_num_instances`.
          include_mask: a bool to indicate whether parse mask groundtruth.
          mask_crop_size: the size which groundtruth mask is cropped to.
        """

        self._max_num_instances = config.maskrcnn_parser.get("max_num_instances", 100)
        self._skip_crowd_during_training = config.maskrcnn_parser.get("skip_crowd_during_training", True)
        self._is_training = is_train
        self._global_batch_size = config.batch_size
        self._num_preprocess_workers = config.get("workers", tf.data.experimental.AUTOTUNE)

        # Anchor
        self._output_size = config.input_info.sample_size[1:3]
        self._min_level = config.model_params.architecture.min_level
        self._max_level = config.model_params.architecture.max_level
        self._num_scales = config.anchor.num_scales
        self._aspect_ratios = config.anchor.aspect_ratios
        self._anchor_size = config.anchor.anchor_size

        # Target assigning
        self._rpn_match_threshold = config.maskrcnn_parser.get("rpn_match_threshold", 0.7)
        self._rpn_unmatched_threshold = config.maskrcnn_parser.get("rpn_unmatched_threshold", 0.3)
        self._rpn_batch_size_per_im = config.maskrcnn_parser.get("rpn_batch_size_per_im", 256)
        self._rpn_fg_fraction = config.maskrcnn_parser.get("rpn_fg_fraction", 0.5)

        # Data augmentation.
        self._aug_rand_hflip = config.maskrcnn_parser.get("aug_rand_hflip", False)
        self._aug_scale_min = config.maskrcnn_parser.get("aug_scale_min", 1.0)
        self._aug_scale_max = config.maskrcnn_parser.get("aug_scale_max", 1.0)

        # Mask
        self._include_mask = config.get("include_mask", False)
        self._mask_crop_size = config.maskrcnn_parser.get("mask_crop_size", 112)

        # Data parsing depends on the `is_training` flag
        if self._is_training:
            self._parse_fn = self._parse_train_data
        else:
            self._parse_fn = self._parse_predict_data

    def create_preprocess_input_fn(self):
        """Parses data to an image and associated training labels.

        Args:
            value: a string tensor holding a serialized tf.Example proto.

        Returns:
            image, labels: if mode == ModeKeys.TRAIN. see _parse_train_data.
            {'images': image, 'labels': labels}: if mode == ModeKeys.PREDICT or ModeKeys.PREDICT_WITH_GT.
        """
        return None, self._pipeline_fn

    def _parse_train_data(self, data):
        """Parses data for training.

        Args:
            data: the decoded tensor dictionary from TfExampleDecoder.

        Returns:
            image: image tensor that is preproessed to have normalized value and
                dimension [output_size[0], output_size[1], 3]
            labels: a dictionary of tensors used for training. The following describes
                {key: value} pairs in the dictionary.
            image_info: a 2D `Tensor` that encodes the information of the image and
                the applied preprocessing. It is in the format of
                [[original_height, original_width], [scaled_height, scaled_width],
            anchor_boxes: ordered dictionary with keys [min_level, min_level+1, ..., max_level].
                The values are tensor with shape [height_l, width_l, 4] representing anchor boxes at each level.
            rpn_score_targets: ordered dictionary with keys [min_level, min_level+1, ..., max_level].
                The values are tensor with shape [height_l, width_l, anchors_per_location]. The height_l and
                width_l represent the dimension of class logits at l-th level.
            rpn_box_targets: ordered dictionary with keys [min_level, min_level+1, ..., max_level].
                The values are tensor with shape [height_l, width_l, anchors_per_location * 4]. The height_l and
                width_l represent the dimension of bounding box regression output at
                l-th level.
            gt_boxes: Groundtruth bounding box annotations. The box is represented in [y1, x1, y2, x2] format.
                The coordinates are w.r.t the scaled image that is fed to the network. The tennsor is
                padded with -1 to the fixed dimension [self._max_num_instances, 4].
            gt_classes: Groundtruth classes annotations. The tennsor is padded with -1 to the fixed
                dimension [self._max_num_instances].
            gt_masks: groundtrugh masks cropped by the bounding box and resized to a fixed size
                determined by mask_crop_size.
        """

        classes = data["groundtruth_classes"]
        boxes = data["groundtruth_boxes"]
        is_crowds = data["groundtruth_is_crowd"]
        if self._include_mask:
            masks = data["groundtruth_instance_masks"]

        # Skips annotations with `is_crowd` = True.
        if self._skip_crowd_during_training and self._is_training:
            num_groundtruths = tf.shape(classes)[0]
            with tf.control_dependencies([num_groundtruths, is_crowds]):
                indices = tf.cond(
                    tf.greater(tf.size(is_crowds), 0),
                    lambda: tf.where(tf.logical_not(is_crowds))[:, 0],
                    lambda: tf.cast(tf.range(num_groundtruths), tf.int64),
                )
            classes = tf.gather(classes, indices, axis=None)
            boxes = tf.gather(boxes, indices, axis=None)
            if self._include_mask:
                masks = tf.gather(masks, indices, axis=None)

        # Gets original image and its size.
        image = data["image"]
        image_shape = tf.shape(image)[0:2]

        # Normalizes image with mean and std pixel values.
        image = input_utils.normalize_image(image)

        # Flips image randomly during training.
        if self._aug_rand_hflip:
            if self._include_mask:
                image, boxes, masks = input_utils.random_horizontal_flip(image, boxes, masks)
            else:
                image, boxes = input_utils.random_horizontal_flip(image, boxes)

        # Converts boxes from normalized coordinates to pixel coordinates.
        # Now the coordinates of boxes are w.r.t. the original image.
        boxes = box_utils.denormalize_boxes(boxes, image_shape)

        # Resizes and crops image.
        image, image_info = input_utils.resize_and_crop_image(
            image,
            self._output_size,
            padded_size=input_utils.compute_padded_size(self._output_size, 2**self._max_level),
            aug_scale_min=self._aug_scale_min,
            aug_scale_max=self._aug_scale_max,
        )
        image_height, image_width, _ = image.get_shape().as_list()

        # Resizes and crops boxes.
        # Now the coordinates of boxes are w.r.t the scaled image.
        image_scale = image_info[2, :]
        offset = image_info[3, :]
        boxes = input_utils.resize_and_crop_boxes(boxes, image_scale, image_info[1, :], offset)

        # Filters out ground truth boxes that are all zeros.
        indices = box_utils.get_non_empty_box_indices(boxes)
        boxes = tf.gather(boxes, indices, axis=None)
        classes = tf.gather(classes, indices, axis=None)

        if self._include_mask:
            masks = tf.gather(masks, indices, axis=None)
            # Transfer boxes to the original image space and do normalization.
            cropped_boxes = boxes + tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
            cropped_boxes /= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
            cropped_boxes = box_utils.normalize_boxes(cropped_boxes, image_shape)
            num_masks = tf.shape(masks)[0]
            masks = tf.image.crop_and_resize(
                tf.expand_dims(masks, axis=-1),
                cropped_boxes,
                box_indices=tf.range(num_masks, dtype=tf.int32),
                crop_size=[self._mask_crop_size, self._mask_crop_size],
                method="bilinear",
            )
            masks = tf.squeeze(masks, axis=-1)

        # Assigns anchor targets.
        # Note that after the target assignment, box targets are absolute pixel
        # offsets w.r.t. the scaled image.
        input_anchor = anchor.Anchor(
            self._min_level,
            self._max_level,
            self._num_scales,
            self._aspect_ratios,
            self._anchor_size,
            (image_height, image_width),
        )

        anchor_labeler = anchor.RpnAnchorLabeler(
            input_anchor,
            self._rpn_match_threshold,
            self._rpn_unmatched_threshold,
            self._rpn_batch_size_per_im,
            self._rpn_fg_fraction,
        )

        rpn_score_targets, rpn_box_targets = anchor_labeler.label_anchors(
            boxes, tf.cast(tf.expand_dims(classes, axis=-1), tf.float32)
        )

        inputs = {
            "image": image,
            "image_info": image_info,
        }

        # Packs labels for model_fn outputs.
        labels = {
            "anchor_boxes": input_anchor.multilevel_boxes,
            "image_info": image_info,
            "rpn_score_targets": rpn_score_targets,
            "rpn_box_targets": rpn_box_targets,
        }

        inputs["gt_boxes"] = input_utils.pad_to_fixed_size(boxes, self._max_num_instances, -1)
        inputs["gt_classes"] = input_utils.pad_to_fixed_size(classes, self._max_num_instances, -1)

        if self._include_mask:
            inputs["gt_masks"] = input_utils.pad_to_fixed_size(masks, self._max_num_instances, -1)

        return inputs, labels

    def _parse_predict_data(self, data):
        """Parses data for prediction.

        Args:
            data: the decoded tensor dictionary from TfExampleDecoder.

        Returns: A dictionary of {'images': image, 'labels': labels} where
            image: image tensor that is preproessed to have normalized value and
                dimension [output_size[0], output_size[1], 3]
            labels: a dictionary of tensors used for training. The following
                describes {key: value} pairs in the dictionary.
            source_ids: Source image id. Default value -1 if the source id is
                empty in the groundtruth annotation.
            image_info: a 2D `Tensor` that encodes the information of the image
                and the applied preprocessing. It is in the format of
                [[original_height, original_width], [scaled_height, scaled_width]].
            anchor_boxes: ordered dictionary with keys
                [min_level, min_level+1, ..., max_level]. The values are tensor with
                shape [height_l, width_l, 4] representing anchor boxes at each level.
        """

        # Gets original image and its size.
        image = data["image"]
        image_shape = tf.shape(image)[0:2]

        # Normalizes image with mean and std pixel values.
        image = input_utils.normalize_image(image)

        # Resizes and crops image.
        image, image_info = input_utils.resize_and_crop_image(
            image,
            self._output_size,
            padded_size=input_utils.compute_padded_size(self._output_size, 2**self._max_level),
            aug_scale_min=1.0,
            aug_scale_max=1.0,
        )

        labels = {
            "image_info": image_info,
        }

        # Converts boxes from normalized coordinates to pixel coordinates.
        boxes = box_utils.denormalize_boxes(data["groundtruth_boxes"], image_shape)
        groundtruths = {
            "source_id": data["source_id"],
            "height": data["height"],
            "width": data["width"],
            "num_detections": tf.shape(data["groundtruth_classes"]),
            "boxes": boxes,
            "classes": data["groundtruth_classes"],
            "areas": data["groundtruth_area"],
            "is_crowds": tf.cast(data["groundtruth_is_crowd"], tf.int32),
        }

        groundtruths["source_id"] = dataloader_utils.process_source_id(groundtruths["source_id"])
        groundtruths = dataloader_utils.pad_groundtruths_to_fixed_size(groundtruths, self._max_num_instances)
        # Remove the `groundtruth` layer key (no longer needed)
        labels["groundtruths"] = groundtruths

        inputs = {
            "image": image,
            "image_info": image_info,
        }

        return inputs, labels

    def _pipeline_fn(self, dataset, decoder_fn):
        preprocess_input_fn = self._parse_fn
        preprocess_pipeline = lambda record: preprocess_input_fn(decoder_fn(record))

        dataset = dataset.map(preprocess_pipeline, num_parallel_calls=self._num_preprocess_workers)
        dataset = dataset.batch(self._global_batch_size, drop_remainder=True)

        return dataset
