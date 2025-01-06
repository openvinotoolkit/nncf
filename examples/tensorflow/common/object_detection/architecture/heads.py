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

import functools

import numpy as np
import tensorflow as tf

from examples.tensorflow.common.object_detection.architecture import nn_ops


class RetinanetHead:
    """RetinaNet head."""

    def __init__(
        self,
        min_level,
        max_level,
        num_classes,
        anchors_per_location,
        num_convs=4,
        num_filters=256,
        use_separable_conv=False,
        norm_activation=nn_ops.norm_activation_builder(activation="relu"),
    ):
        """Initialize params to build RetinaNet head.

        Args:
          min_level: `int` number of minimum feature level.
          max_level: `int` number of maximum feature level.
          num_classes: `int` number of classification categories.
          anchors_per_location: `int` number of anchors per pixel location.
          num_convs: `int` number of stacked convolution before the last prediction
            layer.
          num_filters: `int` number of filters used in the head architecture.
          use_separable_conv: `bool` to indicate whether to use separable
            convoluation.
          norm_activation: an operation that includes a normalization layer followed
            by an optional activation layer.
        """
        self._min_level = min_level
        self._max_level = max_level

        self._num_classes = num_classes
        self._anchors_per_location = anchors_per_location

        self._num_convs = num_convs
        self._num_filters = num_filters
        self._use_separable_conv = use_separable_conv
        with tf.name_scope("class_net") as scope_name:
            self._class_name_scope = tf.name_scope(scope_name)
        with tf.name_scope("box_net") as scope_name:
            self._box_name_scope = tf.name_scope(scope_name)
        self._build_class_net_layers(norm_activation)
        self._build_box_net_layers(norm_activation)

    def _class_net_batch_norm_name(self, i, level):
        return "class-%d-%d" % (i, level)

    def _box_net_batch_norm_name(self, i, level):
        return "box-%d-%d" % (i, level)

    def _build_class_net_layers(self, norm_activation):
        """Build re-usable layers for class prediction network."""
        if self._use_separable_conv:
            self._class_predict = tf.keras.layers.SeparableConv2D(
                self._num_classes * self._anchors_per_location,
                kernel_size=(3, 3),
                bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
                padding="same",
                name="class-predict",
            )
        else:
            self._class_predict = tf.keras.layers.Conv2D(
                self._num_classes * self._anchors_per_location,
                kernel_size=(3, 3),
                bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
                padding="same",
                name="class-predict",
            )

        self._class_conv = []
        self._class_norm_activation = {}
        for i in range(self._num_convs):
            if self._use_separable_conv:
                self._class_conv.append(
                    tf.keras.layers.SeparableConv2D(
                        self._num_filters,
                        kernel_size=(3, 3),
                        bias_initializer=tf.zeros_initializer(),
                        activation=None,
                        padding="same",
                        name="class-" + str(i),
                    )
                )
            else:
                self._class_conv.append(
                    tf.keras.layers.Conv2D(
                        self._num_filters,
                        kernel_size=(3, 3),
                        bias_initializer=tf.zeros_initializer(),
                        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                        activation=None,
                        padding="same",
                        name="class-" + str(i),
                    )
                )

            for level in range(self._min_level, self._max_level + 1):
                name = self._class_net_batch_norm_name(i, level)
                self._class_norm_activation[name] = norm_activation(name=name)

    def _build_box_net_layers(self, norm_activation):
        """Build re-usable layers for box prediction network."""
        if self._use_separable_conv:
            self._box_predict = tf.keras.layers.SeparableConv2D(
                4 * self._anchors_per_location,
                kernel_size=(3, 3),
                bias_initializer=tf.zeros_initializer(),
                padding="same",
                name="box-predict",
            )
        else:
            self._box_predict = tf.keras.layers.Conv2D(
                4 * self._anchors_per_location,
                kernel_size=(3, 3),
                bias_initializer=tf.zeros_initializer(),
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
                padding="same",
                name="box-predict",
            )

        self._box_conv = []
        self._box_norm_activation = {}
        for i in range(self._num_convs):
            if self._use_separable_conv:
                self._box_conv.append(
                    tf.keras.layers.SeparableConv2D(
                        self._num_filters,
                        kernel_size=(3, 3),
                        activation=None,
                        bias_initializer=tf.zeros_initializer(),
                        padding="same",
                        name="box-" + str(i),
                    )
                )
            else:
                self._box_conv.append(
                    tf.keras.layers.Conv2D(
                        self._num_filters,
                        kernel_size=(3, 3),
                        activation=None,
                        bias_initializer=tf.zeros_initializer(),
                        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                        padding="same",
                        name="box-" + str(i),
                    )
                )

            for level in range(self._min_level, self._max_level + 1):
                name = self._box_net_batch_norm_name(i, level)
                self._box_norm_activation[name] = norm_activation(name=name)

    def __call__(self, fpn_features, is_training=None):
        """Returns outputs of RetinaNet head."""
        class_outputs = {}
        box_outputs = {}
        with tf.name_scope("retinanet_head"):
            for level in range(self._min_level, self._max_level + 1):
                features = fpn_features[level]
                class_outputs[str(level)] = self.class_net(features, level, is_training=is_training)
                box_outputs[str(level)] = self.box_net(features, level, is_training=is_training)

        return class_outputs, box_outputs

    def class_net(self, features, level, is_training):
        """Class prediction network for RetinaNet."""
        with self._class_name_scope:
            for i in range(self._num_convs):
                features = self._class_conv[i](features)
                # The convolution layers in the class net are shared among all levels,
                # but each level has its batch normlization to capture the statistical
                # difference among different levels.
                name = self._class_net_batch_norm_name(i, level)
                features = self._class_norm_activation[name](features, is_training=is_training)

            classes = self._class_predict(features)

        return classes

    def box_net(self, features, level, is_training=None):
        """Box regression network for RetinaNet."""
        with self._box_name_scope:
            for i in range(self._num_convs):
                features = self._box_conv[i](features)
                # The convolution layers in the box net are shared among all levels, but
                # each level has its batch normlization to capture the statistical
                # difference among different levels.
                name = self._box_net_batch_norm_name(i, level)
                features = self._box_norm_activation[name](features, is_training=is_training)

            boxes = self._box_predict(features)
        return boxes


class RpnHead(tf.keras.layers.Layer):
    """Region Proposal Network head."""

    def __init__(
        self,
        min_level,
        max_level,
        anchors_per_location,
        num_convs=2,
        num_filters=256,
        use_separable_conv=False,
        activation="relu",
        use_batch_norm=True,
        norm_activation=nn_ops.norm_activation_builder(activation="relu"),
    ):
        """Initialize params to build Region Proposal Network head.

        Args:
            min_level: `int` number of minimum feature level.
            max_level: `int` number of maximum feature level.
            anchors_per_location: `int` number of number of anchors per pixel
                location.
            num_convs: `int` number that represents the number of the intermediate
                conv layers before the prediction.
            num_filters: `int` number that represents the number of filters of the
                intermediate conv layers.
            use_separable_conv: `bool`, indicating whether the separable conv layers
                is used.
            activation: activation function. Support 'relu' and 'swish'.
            use_batch_norm: 'bool', indicating whether batchnorm layers are added.
            norm_activation: an operation that includes a normalization layer followed
                by an optional activation layer.
        """
        super().__init__()
        self._min_level = min_level
        self._max_level = max_level
        self._anchors_per_location = anchors_per_location
        if activation == "relu":
            self._activation_op = tf.nn.relu
        elif activation == "swish":
            self._activation_op = tf.nn.swish
        else:
            raise ValueError("Unsupported activation `{}`.".format(activation))
        self._use_batch_norm = use_batch_norm

        if use_separable_conv:
            self._conv2d_op = functools.partial(
                tf.keras.layers.SeparableConv2D, depth_multiplier=1, bias_initializer=tf.zeros_initializer()
            )
        else:
            self._conv2d_op = functools.partial(
                tf.keras.layers.Conv2D,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer=tf.zeros_initializer(),
            )

        self._rpn_conv = self._conv2d_op(
            num_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=(None if self._use_batch_norm else self._activation_op),
            padding="same",
            name="rpn",
        )
        self._rpn_class_conv = self._conv2d_op(
            anchors_per_location, kernel_size=(1, 1), strides=(1, 1), padding="valid", name="rpn-class"
        )
        self._rpn_box_conv = self._conv2d_op(
            4 * anchors_per_location, kernel_size=(1, 1), strides=(1, 1), padding="valid", name="rpn-box"
        )

        self._norm_activations = {}
        if self._use_batch_norm:
            for level in range(self._min_level, self._max_level + 1):
                self._norm_activations[level] = norm_activation(name="rpn-l%d-bn" % level)

    def _shared_rpn_heads(self, features, anchors_per_location, level, is_training):
        """Shared RPN heads."""
        features = self._rpn_conv(features)
        if self._use_batch_norm:
            # The batch normalization layers are not shared between levels.
            features = self._norm_activations[level](features, is_training=is_training)
        # Proposal classification scores
        scores = self._rpn_class_conv(features)
        # Proposal bbox regression deltas
        bboxes = self._rpn_box_conv(features)

        return scores, bboxes

    def __call__(self, features, is_training=None):
        scores_outputs = {}
        box_outputs = {}
        with tf.name_scope("rpn_head"):
            for level in range(self._min_level, self._max_level + 1):
                scores_output, box_output = self._shared_rpn_heads(
                    features[level], self._anchors_per_location, level, is_training
                )
                scores_outputs[str(level)] = scores_output
                box_outputs[str(level)] = box_output
            return scores_outputs, box_outputs


class FastrcnnHead(tf.keras.layers.Layer):
    """Fast R-CNN box head."""

    def __init__(
        self,
        num_classes,
        num_convs=0,
        num_filters=256,
        use_separable_conv=False,
        num_fcs=2,
        fc_dims=1024,
        activation="relu",
        use_batch_norm=True,
        norm_activation=nn_ops.norm_activation_builder(activation="relu"),
    ):
        """Initialize params to build Fast R-CNN box head.
        Args:
            num_classes: a integer for the number of classes.
            num_convs: `int` number that represents the number of the intermediate
                conv layers before the FC layers.
            num_filters: `int` number that represents the number of filters of the
                intermediate conv layers.
            use_separable_conv: `bool`, indicating whether the separable conv layers
                is used.
            num_fcs: `int` number that represents the number of FC layers before the
                predictions.
            fc_dims: `int` number that represents the number of dimension of the FC
                layers.
            activation: activation function. Support 'relu' and 'swish'.
            use_batch_norm: 'bool', indicating whether batchnorm layers are added.
            norm_activation: an operation that includes a normalization layer followed
                by an optional activation layer.
        """
        super().__init__()
        self._num_classes = num_classes
        self._num_convs = num_convs
        self._num_filters = num_filters

        if use_separable_conv:
            self._conv2d_op = functools.partial(
                tf.keras.layers.SeparableConv2D, depth_multiplier=1, bias_initializer=tf.zeros_initializer()
            )
        else:
            self._conv2d_op = functools.partial(
                tf.keras.layers.Conv2D,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=2, mode="fan_out", distribution="untruncated_normal"
                ),
                bias_initializer=tf.zeros_initializer(),
            )

        self._num_fcs = num_fcs
        self._fc_dims = fc_dims
        if activation == "relu":
            self._activation_op = tf.nn.relu
        elif activation == "swish":
            self._activation_op = tf.nn.swish
        else:
            raise ValueError("Unsupported activation `{}`.".format(activation))
        self._use_batch_norm = use_batch_norm
        self._norm_activation = norm_activation

        self._conv_ops = []
        self._conv_bn_ops = []
        for i in range(self._num_convs):
            self._conv_ops.append(
                self._conv2d_op(
                    self._num_filters,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    dilation_rate=(1, 1),
                    activation=(None if self._use_batch_norm else self._activation_op),
                    name="conv_{}".format(i),
                )
            )
            if self._use_batch_norm:
                self._conv_bn_ops.append(self._norm_activation())

        self._fc_ops = []
        self._fc_bn_ops = []
        for i in range(self._num_fcs):
            self._fc_ops.append(
                tf.keras.layers.Dense(
                    units=self._fc_dims,
                    activation=(None if self._use_batch_norm else self._activation_op),
                    name="fc{}".format(i),
                )
            )
            if self._use_batch_norm:
                self._fc_bn_ops.append(self._norm_activation(fused=False))

        self._class_predict = tf.keras.layers.Dense(
            self._num_classes,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer=tf.zeros_initializer(),
            name="class-predict",
        )
        self._box_predict = tf.keras.layers.Dense(
            self._num_classes * 4,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
            bias_initializer=tf.zeros_initializer(),
            name="box-predict",
        )

    def __call__(self, roi_features, is_training=None):
        """Box and class branches for the Mask-RCNN model.

        Args:
            roi_features: A ROI feature tensor of shape [batch_size, num_rois,
                height_l, width_l, num_filters].
            is_training: `boolean`, if True if model is in training mode.

        Returns:
            class_outputs: a tensor with a shape of
                [batch_size, num_rois, num_classes], representing the class predictions.
            box_outputs: a tensor with a shape of
                [batch_size, num_rois, num_classes * 4], representing the box
                predictions.
        """

        with tf.name_scope("fast_rcnn_head"):
            # reshape inputs beofre FC.
            _, num_rois, height, width, filters = roi_features.get_shape().as_list()

            net = tf.reshape(roi_features, [-1, height, width, filters])
            for i in range(self._num_convs):
                net = self._conv_ops[i](net)
                if self._use_batch_norm:
                    net = self._conv_bn_ops[i](net, is_training=is_training)

            filters = self._num_filters if self._num_convs > 0 else filters
            net = tf.reshape(net, [-1, num_rois, height * width * filters])

            for i in range(self._num_fcs):
                net = self._fc_ops[i](net)
                if self._use_batch_norm:
                    net = self._fc_bn_ops[i](net, is_training=is_training)

            class_outputs = self._class_predict(net)
            box_outputs = self._box_predict(net)
            return class_outputs, box_outputs


class MaskrcnnHead(tf.keras.layers.Layer):
    """Mask R-CNN head."""

    def __init__(
        self,
        num_classes,
        mask_target_size,
        num_convs=4,
        num_filters=256,
        use_separable_conv=False,
        activation="relu",
        use_batch_norm=True,
        norm_activation=nn_ops.norm_activation_builder(activation="relu"),
    ):
        """Initialize params to build Fast R-CNN head.

        Args:
            num_classes: a integer for the number of classes.
            mask_target_size: a integer that is the resolution of masks.
            num_convs: `int` number that represents the number of the intermediate
                conv layers before the prediction.
            num_filters: `int` number that represents the number of filters of the
                intermediate conv layers.
            use_separable_conv: `bool`, indicating whether the separable conv layers
                is used.
            activation: activation function. Support 'relu' and 'swish'.
            use_batch_norm: 'bool', indicating whether batchnorm layers are added.
            norm_activation: an operation that includes a normalization layer followed
                by an optional activation layer.
        """
        super().__init__()
        self._num_classes = num_classes
        self._mask_target_size = mask_target_size

        self._num_convs = num_convs
        self._num_filters = num_filters
        if use_separable_conv:
            self._conv2d_op = functools.partial(
                tf.keras.layers.SeparableConv2D, depth_multiplier=1, bias_initializer=tf.zeros_initializer()
            )
        else:
            self._conv2d_op = functools.partial(
                tf.keras.layers.Conv2D,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=2, mode="fan_out", distribution="untruncated_normal"
                ),
                bias_initializer=tf.zeros_initializer(),
            )
        if activation == "relu":
            self._activation_op = tf.nn.relu
        elif activation == "swish":
            self._activation_op = tf.nn.swish
        else:
            raise ValueError("Unsupported activation `{}`.".format(activation))
        self._use_batch_norm = use_batch_norm
        self._norm_activation = norm_activation
        self._conv2d_ops = []
        for i in range(self._num_convs):
            self._conv2d_ops.append(
                self._conv2d_op(
                    self._num_filters,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    dilation_rate=(1, 1),
                    activation=(None if self._use_batch_norm else self._activation_op),
                    name="mask-conv-l%d" % i,
                )
            )

        self._mask_conv_transpose = tf.keras.layers.Conv2DTranspose(
            self._num_filters,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="valid",
            activation=(None if self._use_batch_norm else self._activation_op),
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2, mode="fan_out", distribution="untruncated_normal"
            ),
            bias_initializer=tf.zeros_initializer(),
            name="conv5-mask",
        )

    def __call__(self, roi_features, class_indices, is_training=None):
        """Mask branch for the Mask-RCNN model.

        Args:
            roi_features: A ROI feature tensor of shape [batch_size, num_rois,
                height_l, width_l, num_filters].
            class_indices: a Tensor of shape [batch_size, num_rois], indicating which
                class the ROI is.
            is_training: `boolean`, if True if model is in training mode.

        Returns:
            mask_outputs: a tensor with a shape of
                [batch_size, num_masks, mask_height, mask_width, num_classes],
                representing the mask predictions.
            fg_gather_indices: a tensor with a shape of [batch_size, num_masks, 2],
                representing the fg mask targets.
        Raises:
          ValueError: If boxes is not a rank-3 tensor or the last dimension of
            boxes is not 4.
        """

        with tf.name_scope("mask_head"):
            _, num_rois, height, width, filters = roi_features.get_shape().as_list()
            net = tf.reshape(roi_features, [-1, height, width, filters])

            for i in range(self._num_convs):
                net = self._conv2d_ops[i](net)
                if self._use_batch_norm:
                    net = self._norm_activation()(net, is_training=is_training)

            net = self._mask_conv_transpose(net)
            if self._use_batch_norm:
                net = self._norm_activation()(net, is_training=is_training)

            mask_outputs = self._conv2d_op(
                self._num_classes, kernel_size=(1, 1), strides=(1, 1), padding="valid", name="mask_fcn_logits"
            )(net)
            mask_outputs = tf.reshape(
                mask_outputs, [-1, num_rois, self._mask_target_size, self._mask_target_size, self._num_classes]
            )

            with tf.name_scope("masks_post_processing"):
                batch_size, num_masks = class_indices.get_shape().as_list()
                mask_outputs = tf.transpose(a=mask_outputs, perm=[0, 1, 4, 2, 3])
                # Contructs indices for gather.
                batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, num_masks])
                mask_indices = tf.tile(tf.expand_dims(tf.range(num_masks), axis=0), [batch_size, 1])
                gather_indices = tf.stack([batch_indices, mask_indices, class_indices], axis=2)
                mask_outputs = tf.gather_nd(mask_outputs, gather_indices)
        return mask_outputs


class YOLOv4:
    """YOLOv4 neck and head"""

    def DarknetConv2D_BN_Leaky(self, *args, **kwargs):
        """Darknet Convolution2D followed by SyncBatchNormalization and LeakyReLU."""
        no_bias_kwargs = {"use_bias": False}
        no_bias_kwargs.update(kwargs)
        return nn_ops.compose(
            nn_ops.DarknetConv2D(*args, **no_bias_kwargs),
            tf.keras.layers.experimental.SyncBatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
        )

    def Spp_Conv2D_BN_Leaky(self, x, num_filters):
        y1 = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding="same")(x)
        y2 = tf.keras.layers.MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding="same")(x)
        y3 = tf.keras.layers.MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding="same")(x)

        y = nn_ops.compose(tf.keras.layers.Concatenate(), self.DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(
            [y3, y2, y1, x]
        )
        return y

    def make_yolo_head(self, x, num_filters):
        """6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer"""
        x = nn_ops.compose(
            self.DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
            self.DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
            self.DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
            self.DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
            self.DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        )(x)

        return x

    def make_yolo_spp_head(self, x, num_filters):
        """6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer"""
        x = nn_ops.compose(
            self.DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
            self.DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
            self.DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        )(x)

        x = self.Spp_Conv2D_BN_Leaky(x, num_filters)

        x = nn_ops.compose(
            self.DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)), self.DarknetConv2D_BN_Leaky(num_filters, (1, 1))
        )(x)

        return x

    def __call__(self, feature_maps, feature_channel_nums, num_anchors, num_classes):
        f1, f2, f3 = feature_maps
        f1_channel_num, f2_channel_num, f3_channel_num = feature_channel_nums

        # feature map 1 head (19x19 for 608 input)
        x1 = self.make_yolo_spp_head(f1, f1_channel_num // 2)

        # upsample fpn merge for feature map 1 & 2
        x1_upsample = nn_ops.compose(
            self.DarknetConv2D_BN_Leaky(f2_channel_num // 2, (1, 1)), tf.keras.layers.UpSampling2D(2)
        )(x1)

        x2 = self.DarknetConv2D_BN_Leaky(f2_channel_num // 2, (1, 1))(f2)
        x2 = tf.keras.layers.Concatenate()([x2, x1_upsample])

        # feature map 2 head (38x38 for 608 input)
        x2 = self.make_yolo_head(x2, f2_channel_num // 2)

        # upsample fpn merge for feature map 2 & 3
        x2_upsample = nn_ops.compose(
            self.DarknetConv2D_BN_Leaky(f3_channel_num // 2, (1, 1)), tf.keras.layers.UpSampling2D(2)
        )(x2)

        x3 = self.DarknetConv2D_BN_Leaky(f3_channel_num // 2, (1, 1))(f3)
        x3 = tf.keras.layers.Concatenate()([x3, x2_upsample])

        # feature map 3 head & output (76x76 for 608 input)
        # x3, y3 = make_last_layers(x3, f3_channel_num//2, num_anchors*(num_classes+5))
        x3 = self.make_yolo_head(x3, f3_channel_num // 2)
        y3 = nn_ops.compose(
            self.DarknetConv2D_BN_Leaky(f3_channel_num, (3, 3)),
            nn_ops.DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name="predict_conv_3"),
        )(x3)

        # downsample fpn merge for feature map 3 & 2
        x3_downsample = nn_ops.compose(
            tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0))),
            self.DarknetConv2D_BN_Leaky(f2_channel_num // 2, (3, 3), strides=(2, 2)),
        )(x3)

        x2 = tf.keras.layers.Concatenate()([x3_downsample, x2])

        # feature map 2 output (38x38 for 608 input)
        # x2, y2 = make_last_layers(x2, 256, num_anchors*(num_classes+5))
        x2 = self.make_yolo_head(x2, f2_channel_num // 2)
        y2 = nn_ops.compose(
            self.DarknetConv2D_BN_Leaky(f2_channel_num, (3, 3)),
            nn_ops.DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name="predict_conv_2"),
        )(x2)

        # downsample fpn merge for feature map 2 & 1
        x2_downsample = nn_ops.compose(
            tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0))),
            self.DarknetConv2D_BN_Leaky(f1_channel_num // 2, (3, 3), strides=(2, 2)),
        )(x2)

        x1 = tf.keras.layers.Concatenate()([x2_downsample, x1])

        # feature map 1 output (19x19 for 608 input)
        # x1, y1 = make_last_layers(x1, f1_channel_num//2, num_anchors*(num_classes+5))
        x1 = self.make_yolo_head(x1, f1_channel_num // 2)
        y1 = nn_ops.compose(
            self.DarknetConv2D_BN_Leaky(f1_channel_num, (3, 3)),
            nn_ops.DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name="predict_conv_1"),
        )(x1)

        return y1, y2, y3
