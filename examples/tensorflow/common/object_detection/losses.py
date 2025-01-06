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

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from examples.tensorflow.common.logger import logger


def focal_loss(logits, targets, alpha, gamma, normalizer):
    """Compute the focal loss between `logits` and the golden `target` values.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.

    Args:
        logits: A float32 tensor of size [batch, height_in, width_in, num_predictions].
        targets: A float32 tensor of size [batch, height_in, width_in, num_predictions].
        alpha: A float32 scalar multiplying alpha to the loss from positive examples
            and (1-alpha) to the loss from negative examples.
        gamma: A float32 scalar modulating loss from hard and easy examples.
        normalizer: A float32 scalar normalizes the total loss from all examples.

    Returns:
        loss: A float32 Tensor of size [batch, height_in, width_in, num_predictions]
        representing normalized loss on the prediction map.
    """

    with tf.name_scope("focal_loss"):
        positive_label_mask = tf.math.equal(targets, 1.0)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)

        # Below are comments/derivations for computing modulator.
        # For brevity, let x = logits,  z = targets, r = gamma, and p_t = sigmod(x)
        # for positive samples and 1 - sigmoid(x) for negative examples.
        #
        # The modulator, defined as (1 - P_t)^r, is a critical part in focal loss
        # computation. For r > 0, it puts more weights on hard examples, and less
        # weights on easier ones. However if it is directly computed as (1 - P_t)^r,
        # its back-propagation is not stable when r < 1. The implementation here
        # resolves the issue.
        #
        # For positive samples (labels being 1),
        #    (1 - p_t)^r
        #  = (1 - sigmoid(x))^r
        #  = (1 - (1 / (1 + exp(-x))))^r
        #  = (exp(-x) / (1 + exp(-x)))^r
        #  = exp(log((exp(-x) / (1 + exp(-x)))^r))
        #  = exp(r * log(exp(-x)) - r * log(1 + exp(-x)))
        #  = exp(- r * x - r * log(1 + exp(-x)))
        #
        # For negative samples (labels being 0),
        #    (1 - p_t)^r
        #  = (sigmoid(x))^r
        #  = (1 / (1 + exp(-x)))^r
        #  = exp(log((1 / (1 + exp(-x)))^r))
        #  = exp(-r * log(1 + exp(-x)))
        #
        # Therefore one unified form for positive (z = 1) and negative (z = 0)
        # samples is:
        #      (1 - p_t)^r = exp(-r * z * x - r * log(1 + exp(-x))).

        neg_logits = -1.0 * logits
        modulator = tf.math.exp(gamma * targets * neg_logits - gamma * tf.math.log1p(tf.math.exp(neg_logits)))
        loss = modulator * cross_entropy
        weighted_loss = tf.where(positive_label_mask, alpha * loss, (1.0 - alpha) * loss)
        weighted_loss /= normalizer

    return weighted_loss


class RpnScoreLoss:
    """Region Proposal Network score loss function."""

    def __init__(self, params):
        self._rpn_batch_size_per_im = params.rpn_batch_size_per_im
        self._binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM, from_logits=True
        )

    def __call__(self, score_outputs, labels):
        """Computes total RPN detection loss.
        Computes total RPN detection loss including box and score from all levels.

        Args:
            score_outputs: an OrderDict with keys representing levels and values
                representing scores in [batch_size, height, width, num_anchors].
            labels: the dictionary that returned from dataloader that includes
                groundturth targets.
        Returns:
            rpn_score_loss: a scalar tensor representing total score loss.
        """

        with tf.name_scope("rpn_loss"):
            levels = sorted(score_outputs.keys())

            score_losses = []
            for level in levels:
                score_losses.append(
                    self._rpn_score_loss(
                        score_outputs[level],
                        labels[int(level)],
                        normalizer=tf.cast(tf.shape(score_outputs[level])[0] * self._rpn_batch_size_per_im, tf.float32),
                    )
                )

            # Sums per level losses to total loss.
            return tf.math.add_n(score_losses)

    def _rpn_score_loss(self, score_outputs, score_targets, normalizer=1.0):
        """Computes score loss.

        score_targets has three values:
            (1) score_targets[i]=1, the anchor is a positive sample.
            (2) score_targets[i]=0, negative.
            (3) score_targets[i]=-1, the anchor is don't care (ignore).
        """

        with tf.name_scope("rpn_score_loss"):
            mask = tf.math.logical_or(tf.math.equal(score_targets, 1), tf.math.equal(score_targets, 0))

            score_targets = tf.math.maximum(score_targets, tf.zeros_like(score_targets))

            score_targets = tf.expand_dims(score_targets, axis=-1)
            score_outputs = tf.expand_dims(score_outputs, axis=-1)
            score_loss = self._binary_crossentropy(score_targets, score_outputs, sample_weight=mask)

            score_loss /= normalizer
            return score_loss


class RpnBoxLoss:
    """Region Proposal Network box regression loss function."""

    def __init__(self, params):
        logger.info("RpnBoxLoss huber_loss_delta {}".format(params.huber_loss_delta))
        # The delta is typically around the mean value of regression target.
        # for instances, the regression targets of 512x512 input with 6 anchors on
        # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].
        self._huber_loss = tf.keras.losses.Huber(delta=params.huber_loss_delta, reduction=tf.keras.losses.Reduction.SUM)

    def __call__(self, box_outputs, labels):
        """Computes total RPN detection loss.
        Computes total RPN detection loss including box and score from all levels.

        Args:
            box_outputs: an OrderDict with keys representing levels and values
                representing box regression targets in
                [batch_size, height, width, num_anchors * 4].
            labels: the dictionary that returned from dataloader that includes
                groundturth targets.

        Returns:
            rpn_box_loss: a scalar tensor representing total box regression loss.
        """

        with tf.name_scope("rpn_loss"):
            levels = sorted(box_outputs.keys())

            box_losses = []
            for level in levels:
                box_losses.append(self._rpn_box_loss(box_outputs[level], labels[int(level)]))

            # Sum per level losses to total loss.
            return tf.add_n(box_losses)

    def _rpn_box_loss(self, box_outputs, box_targets, normalizer=1.0):
        """Computes box regression loss."""
        with tf.name_scope("rpn_box_loss"):
            mask = tf.cast(tf.not_equal(box_targets, 0.0), tf.float32)
            box_targets = tf.expand_dims(box_targets, axis=-1)
            box_outputs = tf.expand_dims(box_outputs, axis=-1)
            box_loss = self._huber_loss(box_targets, box_outputs, sample_weight=mask)
            # The loss is normalized by the sum of non-zero weights and additional
            # normalizer provided by the function caller. Using + 0.01 here to avoid
            # division by zero.
            box_loss /= normalizer * (tf.reduce_sum(mask) + 0.01)
            return box_loss


class FastrcnnClassLoss:
    """Fast R-CNN classification loss function."""

    def __init__(self):
        self._categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM, from_logits=True
        )

    def __call__(self, class_outputs, class_targets):
        """Computes the class loss (Fast-RCNN branch) of Mask-RCNN.
        This function implements the classification loss of the Fast-RCNN.
        The classification loss is softmax on all RoIs.
        Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/fast_rcnn_heads.py

        Args:
            class_outputs: a float tensor representing the class prediction for each box
                with a shape of [batch_size, num_boxes, num_classes].
            class_targets: a float tensor representing the class label for each box
                with a shape of [batch_size, num_boxes].

        Returns:
            a scalar tensor representing total class loss.
        """
        with tf.name_scope("fast_rcnn_loss"):
            batch_size, num_boxes, num_classes = class_outputs.get_shape().as_list()
            class_targets = tf.cast(class_targets, tf.int32)
            class_targets_one_hot = tf.one_hot(class_targets, num_classes, on_value=None, off_value=None)
            return self._fast_rcnn_class_loss(
                class_outputs, class_targets_one_hot, normalizer=batch_size * num_boxes / 2.0
            )

    def _fast_rcnn_class_loss(self, class_outputs, class_targets_one_hot, normalizer):
        """Computes classification loss."""
        with tf.name_scope("fast_rcnn_class_loss"):
            class_loss = self._categorical_crossentropy(class_targets_one_hot, class_outputs)
            class_loss /= normalizer
            return class_loss


class FastrcnnBoxLoss:
    """Fast R-CNN box regression loss function."""

    def __init__(self, params):
        logger.info("FastrcnnBoxLoss huber_loss_delta {}".format(params.huber_loss_delta))
        # The delta is typically around the mean value of regression target.
        # for instances, the regression targets of 512x512 input with 6 anchors on
        # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].
        self._huber_loss = tf.keras.losses.Huber(delta=params.huber_loss_delta, reduction=tf.keras.losses.Reduction.SUM)

    def __call__(self, box_outputs, class_targets, box_targets):
        """Computes the box loss (Fast-RCNN branch) of Mask-RCNN.

        This function implements the box regression loss of the Fast-RCNN. As the
        `box_outputs` produces `num_classes` boxes for each RoI, the reference model
        expands `box_targets` to match the shape of `box_outputs` and selects only
        the target that the RoI has a maximum overlap.
        (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py)
        Instead, this function selects the `box_outputs` by the `class_targets` so
        that it doesn't expand `box_targets`.

        The box loss is smooth L1-loss on only positive samples of RoIs.
        Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/fast_rcnn_heads.py

        Args:
            box_outputs: a float tensor representing the box prediction for each box
                with a shape of [batch_size, num_boxes, num_classes * 4].
            class_targets: a float tensor representing the class label for each box
                with a shape of [batch_size, num_boxes].
            box_targets: a float tensor representing the box label for each box
                with a shape of [batch_size, num_boxes, 4].

        Returns:
            box_loss: a scalar tensor representing total box regression loss.
        """

        with tf.name_scope("fast_rcnn_loss"):
            class_targets = tf.cast(class_targets, tf.int32)

            # Selects the box from `box_outputs` based on `class_targets`, with which
            # the box has the maximum overlap.
            (batch_size, num_rois, num_class_specific_boxes) = box_outputs.get_shape().as_list()
            num_classes = num_class_specific_boxes // 4
            box_outputs = tf.reshape(box_outputs, [batch_size, num_rois, num_classes, 4])

            box_indices = tf.reshape(
                class_targets
                + tf.tile(tf.expand_dims(tf.range(batch_size) * num_rois * num_classes, 1), [1, num_rois])
                + tf.tile(tf.expand_dims(tf.range(num_rois) * num_classes, 0), [batch_size, 1]),
                [-1],
            )

            box_outputs = tf.matmul(
                tf.one_hot(box_indices, batch_size * num_rois * num_classes, None, None, None, box_outputs.dtype),
                tf.reshape(box_outputs, [-1, 4]),
            )
            box_outputs = tf.reshape(box_outputs, [batch_size, -1, 4])

            return self._fast_rcnn_box_loss(box_outputs, box_targets, class_targets)

    def _fast_rcnn_box_loss(self, box_outputs, box_targets, class_targets, normalizer=1.0):
        """Computes box regression loss."""
        with tf.name_scope("fast_rcnn_box_loss"):
            mask = tf.tile(tf.expand_dims(tf.greater(class_targets, 0), axis=2), [1, 1, 4])
            mask = tf.cast(mask, tf.float32)
            box_targets = tf.expand_dims(box_targets, axis=-1)
            box_outputs = tf.expand_dims(box_outputs, axis=-1)
            box_loss = self._huber_loss(box_targets, box_outputs, sample_weight=mask)
            # The loss is normalized by the number of ones in mask,
            # additianal normalizer provided by the user and using 0.01 here to avoid
            # division by 0.
            box_loss /= normalizer * (tf.reduce_sum(mask) + 0.01)
            return box_loss


class MaskrcnnLoss:
    """Mask R-CNN instance segmentation mask loss function."""

    def __init__(self):
        self._binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM, from_logits=True
        )

    def __call__(self, mask_outputs, mask_targets, select_class_targets):
        """Computes the mask loss of Mask-RCNN.

        This function implements the mask loss of Mask-RCNN. As the `mask_outputs`
        produces `num_classes` masks for each RoI, the reference model expands
        `mask_targets` to match the shape of `mask_outputs` and selects only the
        target that the RoI has a maximum overlap.
        (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/mask_rcnn.py)
        Instead, this implementation selects the `mask_outputs` by the `class_targets`
        so that it doesn't expand `mask_targets`. Note that the selection logic is
        done in the post-processing of mask_rcnn_fn in mask_rcnn_architecture.py.

        Args:
            mask_outputs: a float tensor representing the prediction for each mask,
                with a shape of [batch_size, num_masks, mask_height, mask_width].
            mask_targets: a float tensor representing the binary mask of ground truth
                labels for each mask with a shape of [batch_size, num_masks, mask_height, mask_width].
            select_class_targets: a tensor with a shape of [batch_size, num_masks],
                representing the foreground mask targets.

        Returns:
            mask_loss: a float tensor representing total mask loss.
        """

        with tf.name_scope("mask_rcnn_loss"):
            (batch_size, num_masks, mask_height, mask_width) = mask_outputs.get_shape().as_list()
            weights = tf.tile(
                tf.reshape(tf.greater(select_class_targets, 0), [batch_size, num_masks, 1, 1]),
                [1, 1, mask_height, mask_width],
            )
            weights = tf.cast(weights, tf.float32)

            mask_targets = tf.expand_dims(mask_targets, axis=-1)
            mask_outputs = tf.expand_dims(mask_outputs, axis=-1)
            mask_loss = self._binary_crossentropy(mask_targets, mask_outputs, sample_weight=weights)

            # The loss is normalized by the number of 1's in weights and
            # + 0.01 is used to avoid division by zero.
            return mask_loss / (tf.reduce_sum(weights) + 0.01)


class RetinanetClassLoss:
    """RetinaNet class loss."""

    def __init__(self, params, num_classes):
        self._num_classes = num_classes
        self._focal_loss_alpha = params.focal_loss_alpha
        self._focal_loss_gamma = params.focal_loss_gamma

    def __call__(self, cls_outputs, labels, num_positives):
        """Computes total detection loss.

        Computes total detection loss including box and class loss from all levels.

        Args:
            cls_outputs: an OrderDict with keys representing levels and values
                representing logits in [batch_size, height, width,
                num_anchors * num_classes].
            labels: the dictionary that returned from dataloader that includes
                class groundturth targets.
            num_positives: number of positive examples in the minibatch.

        Returns:
            an integar tensor representing total class loss.
        """
        # Sums all positives in a batch for normalization and avoids zero
        # num_positives_sum, which would lead to inf loss during training
        num_positives_sum = tf.reduce_sum(input_tensor=num_positives) + 1.0

        cls_losses = []
        for level in cls_outputs:
            cls_losses.append(self.class_loss(cls_outputs[level], labels[int(level)], num_positives_sum))

        # Sums per level losses to total loss.
        return tf.add_n(cls_losses)

    def class_loss(self, cls_outputs, cls_targets, num_positives, ignore_label=-2):
        """Computes RetinaNet classification loss."""
        # Onehot encoding for classification labels.
        cls_targets_one_hot = tf.one_hot(cls_targets, self._num_classes, on_value=None, off_value=None)
        bs, height, width, _, _ = cls_targets_one_hot.get_shape().as_list()
        cls_targets_one_hot = tf.reshape(cls_targets_one_hot, [bs, height, width, -1])

        loss = focal_loss(
            tf.cast(cls_outputs, tf.float32),
            tf.cast(cls_targets_one_hot, tf.float32),
            self._focal_loss_alpha,
            self._focal_loss_gamma,
            num_positives,
        )

        ignore_loss = tf.where(
            tf.equal(cls_targets, ignore_label),
            tf.zeros_like(cls_targets, dtype=tf.float32),
            tf.ones_like(cls_targets, dtype=tf.float32),
        )
        ignore_loss = tf.expand_dims(ignore_loss, -1)
        ignore_loss = tf.tile(ignore_loss, [1, 1, 1, 1, self._num_classes])
        ignore_loss = tf.reshape(ignore_loss, tf.shape(input=loss))

        return tf.reduce_sum(input_tensor=ignore_loss * loss)


class RetinanetBoxLoss:
    """RetinaNet box loss."""

    def __init__(self, params):
        self._huber_loss = tf.keras.losses.Huber(delta=params.huber_loss_delta, reduction=tf.keras.losses.Reduction.SUM)

    def __call__(self, box_outputs, labels, num_positives):
        """Computes box detection loss.

        Computes total detection loss including box and class loss from all levels.

        Args:
            box_outputs: an OrderDict with keys representing levels and values
                representing box regression targets in [batch_size, height, width,
                num_anchors * 4].
            labels: the dictionary that returned from dataloader that includes
                box groundturth targets.
            num_positives: number of positive examples in the minibatch.

        Returns:
            an integer tensor representing total box regression loss.
        """

        # Sums all positives in a batch for normalization and avoids zero
        # num_positives_sum, which would lead to inf loss during training
        num_positives_sum = tf.reduce_sum(input_tensor=num_positives) + 1.0

        box_losses = []
        for level in box_outputs:
            box_targets_l = labels[int(level)]
            box_losses.append(self.box_loss(box_outputs[level], box_targets_l, num_positives_sum))

        # Sums per level losses to total loss.
        return tf.add_n(box_losses)

    def box_loss(self, box_outputs, box_targets, num_positives):
        """Computes RetinaNet box regression loss."""
        # The delta is typically around the mean value of regression target.
        # for instances, the regression targets of 512x512 input with 6 anchors on
        # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
        normalizer = num_positives * 4.0
        mask = tf.cast(tf.not_equal(box_targets, 0.0), tf.float32)
        box_targets = tf.expand_dims(box_targets, axis=-1)
        box_outputs = tf.expand_dims(box_outputs, axis=-1)
        box_loss = self._huber_loss(box_targets, box_outputs, sample_weight=mask)
        box_loss /= normalizer

        return box_loss


class YOLOv4Loss:
    """YOLOv4 loss."""

    def softmax_focal_loss(self, y_true, y_pred, gamma=2.0, alpha=0.25):
        """
        Compute softmax focal loss.
        Reference Paper:
            "Focal Loss for Dense Object Detection"
            https://arxiv.org/abs/1708.02002

        :param y_true: Ground truth targets,
                tensor of shape (?, num_boxes, num_classes).
        :param y_pred: Predicted logits,
                tensor of shape (?, num_boxes, num_classes).
        :param gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
        :param alpha: optional alpha weighting factor to balance positives vs negatives.
        :return softmax_focal_loss: Softmax focal loss, tensor of shape (?, num_boxes).
        """
        y_pred = tf.nn.softmax(y_pred)
        y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)

        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate Focal Loss
        softmax_focal_loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy

        return softmax_focal_loss

    def sigmoid_focal_loss(self, y_true, y_pred, gamma=2.0, alpha=0.25):
        """
        Compute sigmoid focal loss.
        Reference Paper:
            "Focal Loss for Dense Object Detection"
            https://arxiv.org/abs/1708.02002

        :param y_true: Ground truth targets,
                tensor of shape (?, num_boxes, num_classes).
        :param y_pred: Predicted logits,
                tensor of shape (?, num_boxes, num_classes).
        :param gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
        :param alpha: optional alpha weighting factor to balance positives vs negatives.
        :return sigmoid_focal_loss: Sigmoid focal loss, tensor of shape (?, num_boxes).
        """
        sigmoid_loss = K.binary_crossentropy(y_true, y_pred, from_logits=True)

        pred_prob = tf.sigmoid(y_pred)
        p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        alpha_weight_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

        sigmoid_focal_loss = modulating_factor * alpha_weight_factor * sigmoid_loss

        return sigmoid_focal_loss

    def box_iou(self, b1, b2):
        """
        Return iou tensor

        :param b1: tensor, shape=(i1,...,iN, 4), xywh
        :param b2: tensor, shape=(j, 4), xywh
        :return iou: tensor, shape=(i1,...,iN, j)
        """
        # Expand dim to apply broadcasting.
        b1 = K.expand_dims(b1, -2)
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.0
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half

        # Expand dim to apply broadcasting.
        b2 = K.expand_dims(b2, 0)
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.0
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        intersect_mins = K.maximum(b1_mins, b2_mins)
        intersect_maxes = K.minimum(b1_maxes, b2_maxes)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        iou = intersect_area / (b1_area + b2_area - intersect_area + K.epsilon())

        return iou

    def box_giou(self, b_true, b_pred):
        """
        Calculate GIoU loss on anchor boxes
        Reference Paper:
            "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"
            https://arxiv.org/abs/1902.09630

        :param b_true: GT boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        :param b_pred: predict boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        :return giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        b_true_xy = b_true[..., :2]
        b_true_wh = b_true[..., 2:4]
        b_true_wh_half = b_true_wh / 2.0
        b_true_mins = b_true_xy - b_true_wh_half
        b_true_maxes = b_true_xy + b_true_wh_half

        b_pred_xy = b_pred[..., :2]
        b_pred_wh = b_pred[..., 2:4]
        b_pred_wh_half = b_pred_wh / 2.0
        b_pred_mins = b_pred_xy - b_pred_wh_half
        b_pred_maxes = b_pred_xy + b_pred_wh_half

        intersect_mins = K.maximum(b_true_mins, b_pred_mins)
        intersect_maxes = K.minimum(b_true_maxes, b_pred_maxes)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b_true_area = b_true_wh[..., 0] * b_true_wh[..., 1]
        b_pred_area = b_pred_wh[..., 0] * b_pred_wh[..., 1]
        union_area = b_true_area + b_pred_area - intersect_area
        # calculate IoU, add epsilon in denominator to avoid dividing by 0
        iou = intersect_area / (union_area + K.epsilon())

        # get enclosed area
        enclose_mins = K.minimum(b_true_mins, b_pred_mins)
        enclose_maxes = K.maximum(b_true_maxes, b_pred_maxes)
        enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        # calculate GIoU, add epsilon in denominator to avoid dividing by 0
        giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + K.epsilon())
        giou = K.expand_dims(giou, -1)

        return giou

    def box_diou(self, b_true, b_pred, use_ciou=False):
        """
        Calculate DIoU/CIoU loss on anchor boxes
        Reference Paper:
            "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
            https://arxiv.org/abs/1911.08287

        :param b_true: GT boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        :param b_pred: predict boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        :param use_ciou: bool flag to indicate whether to use CIoU loss type
        :return diou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        b_true_xy = b_true[..., :2]
        b_true_wh = b_true[..., 2:4]
        b_true_wh_half = b_true_wh / 2.0
        b_true_mins = b_true_xy - b_true_wh_half
        b_true_maxes = b_true_xy + b_true_wh_half

        b_pred_xy = b_pred[..., :2]
        b_pred_wh = b_pred[..., 2:4]
        b_pred_wh_half = b_pred_wh / 2.0
        b_pred_mins = b_pred_xy - b_pred_wh_half
        b_pred_maxes = b_pred_xy + b_pred_wh_half

        intersect_mins = K.maximum(b_true_mins, b_pred_mins)
        intersect_maxes = K.minimum(b_true_maxes, b_pred_maxes)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b_true_area = b_true_wh[..., 0] * b_true_wh[..., 1]
        b_pred_area = b_pred_wh[..., 0] * b_pred_wh[..., 1]
        union_area = b_true_area + b_pred_area - intersect_area
        # calculate IoU, add epsilon in denominator to avoid dividing by 0
        iou = intersect_area / (union_area + K.epsilon())

        # box center distance
        center_distance = K.sum(K.square(b_true_xy - b_pred_xy), axis=-1)
        # get enclosed area
        enclose_mins = K.minimum(b_true_mins, b_pred_mins)
        enclose_maxes = K.maximum(b_true_maxes, b_pred_maxes)
        enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
        # get enclosed diagonal distance
        enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
        # calculate DIoU, add epsilon in denominator to avoid dividing by 0
        diou = iou - 1.0 * (center_distance) / (enclose_diagonal + K.epsilon())
        diou = K.expand_dims(diou, -1)
        return diou

    def _smooth_labels(self, y_true, label_smoothing):
        label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
        return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    def yolo3_decode(self, feats, anchors, num_classes, input_shape, scale_x_y=None):
        """Decode final layer features to bounding box parameters."""
        num_anchors = len(anchors)
        # Reshape to batch, height, width, num_anchors, box_params.
        anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

        grid_shape = K.shape(feats)[1:3]  # height, width
        grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
        grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
        grid = K.concatenate([grid_x, grid_y])
        grid = K.cast(grid, K.dtype(feats))

        feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

        # Adjust preditions to each spatial grid point and anchor size.
        if scale_x_y:
            # Eliminate grid sensitivity trick involved in YOLOv4
            #
            # Reference Paper & code:
            #     "YOLOv4: Optimal Speed and Accuracy of Object Detection"
            #     https://arxiv.org/abs/2004.10934
            #     https://github.com/opencv/opencv/issues/17148
            #
            box_xy_tmp = K.sigmoid(feats[..., :2]) * scale_x_y - (scale_x_y - 1) / 2
            box_xy = (box_xy_tmp + grid) / (K.cast(grid_shape[..., ::-1], K.dtype(feats)) + K.epsilon())
        else:
            box_xy = (K.sigmoid(feats[..., :2]) + grid) / (K.cast(grid_shape[..., ::-1], K.dtype(feats)) + K.epsilon())
        box_wh = (
            K.exp(feats[..., 2:4]) * anchors_tensor / (K.cast(input_shape[..., ::-1], K.dtype(feats)) + K.epsilon())
        )

        return feats, box_xy, box_wh

    def get_anchors(self, anchors_path):
        """loads the anchors from a file"""
        with open(anchors_path, encoding="utf8") as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(",")]
        return np.array(anchors).reshape(-1, 2)

    def __call__(
        self,
        labels,
        outputs,
        anchors,
        num_classes,
        ignore_thresh=0.5,
        label_smoothing=0,
        elim_grid_sense=True,
        use_focal_loss=False,
        use_focal_obj_loss=False,
        use_softmax_loss=False,
        use_giou_loss=False,
        use_diou_loss=True,
    ):
        """
        YOLOv3 loss function.

        :param yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
        :param y_true: list of array, the output of preprocess_true_boxes
        :param anchors: array, shape=(N, 2), wh
        :param num_classes: integer
        :param ignore_thresh: float, the iou threshold whether to ignore object confidence loss
        :return loss: tensor, shape=(1,)
        """
        anchors = np.array(anchors).astype(float).reshape(-1, 2)
        num_layers = len(anchors) // 3  # default setting
        yolo_outputs = list(outputs.values())  # args[:num_layers]
        y_true = list(labels.values())  # args[num_layers:]

        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        scale_x_y = [1.05, 1.1, 1.2] if elim_grid_sense else [None, None, None]

        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
        loss = 0
        total_location_loss = 0
        total_confidence_loss = 0
        total_class_loss = 0
        batch_size = K.shape(yolo_outputs[0])[0]  # batch size, tensor
        batch_size_f = K.cast(batch_size, K.dtype(yolo_outputs[0]))

        for i in range(num_layers):
            object_mask = y_true[i][..., 4:5]
            true_class_probs = y_true[i][..., 5:]
            if label_smoothing:
                true_class_probs = self._smooth_labels(true_class_probs, label_smoothing)
                true_objectness_probs = self._smooth_labels(object_mask, label_smoothing)
            else:
                true_objectness_probs = object_mask

            raw_pred, pred_xy, pred_wh = self.yolo3_decode(
                yolo_outputs[i], anchors[anchor_mask[i]], num_classes, input_shape, scale_x_y=scale_x_y[i]
            )
            pred_box = K.concatenate([pred_xy, pred_wh])

            box_loss_scale = 2 - y_true[i][..., 2:3] * y_true[i][..., 3:4]

            # Find ignore mask, iterate over each of batch.
            ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
            object_mask_bool = K.cast(object_mask, "bool")

            def loop_body(b, ignore_mask):
                true_box = tf.boolean_mask(y_true[i][b, ..., 0:4], object_mask_bool[b, ..., 0])
                iou = self.box_iou(pred_box[b], true_box)
                best_iou = K.max(iou, axis=-1)
                ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
                return b + 1, ignore_mask

            _, ignore_mask = tf.while_loop(lambda b, *args: b < batch_size, loop_body, [0, ignore_mask])
            ignore_mask = ignore_mask.stack()
            ignore_mask = K.expand_dims(ignore_mask, -1)

            raw_pred = raw_pred + K.epsilon()
            if use_focal_obj_loss:
                # Focal loss for objectness confidence
                confidence_loss = self.sigmoid_focal_loss(true_objectness_probs, raw_pred[..., 4:5])
            else:
                confidence_loss = (
                    object_mask * K.binary_crossentropy(true_objectness_probs, raw_pred[..., 4:5], from_logits=True)
                ) + (
                    (1 - object_mask)
                    * ignore_mask
                    * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True)
                )

            if use_focal_loss:
                # Focal loss for classification score
                if use_softmax_loss:
                    class_loss = self.softmax_focal_loss(true_class_probs, raw_pred[..., 5:])
                else:
                    class_loss = self.sigmoid_focal_loss(true_class_probs, raw_pred[..., 5:])
            else:
                if use_softmax_loss:
                    # use softmax style classification output
                    class_loss = object_mask * K.expand_dims(
                        K.categorical_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True), axis=-1
                    )
                else:
                    # use sigmoid style classification output
                    class_loss = object_mask * K.binary_crossentropy(
                        true_class_probs, raw_pred[..., 5:], from_logits=True
                    )

            raw_true_box = y_true[i][..., 0:4]
            diou = self.box_diou(raw_true_box, pred_box)
            diou_loss = object_mask * box_loss_scale * (1 - diou)
            diou_loss = K.sum(diou_loss) / batch_size_f
            location_loss = diou_loss

            confidence_loss = K.sum(confidence_loss) / batch_size_f
            class_loss = K.sum(class_loss) / batch_size_f
            loss += location_loss + confidence_loss + class_loss
            total_location_loss += location_loss
            total_confidence_loss += confidence_loss
            total_class_loss += class_loss

        loss = K.expand_dims(loss, axis=-1)

        return loss, total_location_loss, total_confidence_loss, total_class_loss
