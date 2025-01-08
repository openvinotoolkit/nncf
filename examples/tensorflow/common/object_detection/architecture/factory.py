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

from examples.tensorflow.common.object_detection.architecture import darknet
from examples.tensorflow.common.object_detection.architecture import fpn
from examples.tensorflow.common.object_detection.architecture import heads
from examples.tensorflow.common.object_detection.architecture import nn_ops
from examples.tensorflow.common.object_detection.architecture import resnet


def norm_activation_generator(params):
    return nn_ops.norm_activation_builder(
        momentum=params.batch_norm_momentum,
        epsilon=params.batch_norm_epsilon,
        trainable=True,
        activation=params.activation,
    )


def backbone_generator(params):
    """Generator function for various backbone models."""
    backbone_name = params.model_params.architecture.backbone.name
    if params.model in ("RetinaNet", "MaskRCNN"):
        if backbone_name == "resnet":
            resnet_params = params.model_params.architecture.backbone.params
            backbone_fn = resnet.Resnet(
                resnet_depth=resnet_params.depth,
                activation=params.model_params.norm_activation.activation,
                norm_activation=norm_activation_generator(params.model_params.norm_activation),
            )
        else:
            raise ValueError("Backbone {} is not supported for {} model.".format(backbone_name, params.model))
    elif params.model == "YOLOv4":
        if backbone_name == "darknet":
            backbone_fn = darknet.CSPDarknet53()
        else:
            raise ValueError("Backbone {} is not supported for {} model.".format(backbone_name, params.model))
    else:
        raise ValueError("Model {} is not supported.".format(params.model))

    return backbone_fn


def multilevel_features_generator(params):
    """Generator function for various FPN models."""
    assert params.model_params.architecture.multilevel_features == "fpn"
    fpn_params = params.model_params.architecture.fpn_params
    fpn_fn = fpn.Fpn(
        min_level=params.model_params.architecture.min_level,
        max_level=params.model_params.architecture.max_level,
        fpn_feat_dims=fpn_params.fpn_feat_dims,
        use_separable_conv=fpn_params.use_separable_conv,
        activation=params.model_params.norm_activation.activation,
        use_batch_norm=fpn_params.use_batch_norm,
        norm_activation=norm_activation_generator(params.model_params.norm_activation),
    )

    return fpn_fn


def retinanet_head_generator(params):
    """Generator function for RetinaNet head architecture."""
    head_params = params.model_params.architecture.head_params
    anchors_per_location = params.model_params.anchor.num_scales * len(params.model_params.anchor.aspect_ratios)
    return heads.RetinanetHead(
        params.model_params.architecture.min_level,
        params.model_params.architecture.max_level,
        params.model_params.architecture.num_classes,
        anchors_per_location,
        head_params.num_convs,
        head_params.num_filters,
        head_params.use_separable_conv,
        norm_activation=norm_activation_generator(params.model_params.norm_activation),
    )


def rpn_head_generator(params):
    """Generator function for RPN head architecture."""
    head_params = params.rpn_head
    anchors_per_location = params.anchor.num_scales * len(params.anchor.aspect_ratios)
    return heads.RpnHead(
        params.model_params.architecture.min_level,
        params.model_params.architecture.max_level,
        anchors_per_location,
        head_params.num_convs,
        head_params.num_filters,
        head_params.use_separable_conv,
        params.model_params.norm_activation.activation,
        head_params.use_batch_norm,
        norm_activation=norm_activation_generator(params.model_params.norm_activation),
    )


def fast_rcnn_head_generator(params):
    """Generator function for Fast R-CNN head architecture."""
    head_params = params.frcnn_head
    return heads.FastrcnnHead(
        params.model_params.architecture.num_classes,
        head_params.num_convs,
        head_params.num_filters,
        head_params.use_separable_conv,
        head_params.num_fcs,
        head_params.fc_dims,
        params.model_params.norm_activation.activation,
        head_params.use_batch_norm,
        norm_activation=norm_activation_generator(params.model_params.norm_activation),
    )


def mask_rcnn_head_generator(params):
    """Generator function for Mask R-CNN head architecture."""
    head_params = params.mrcnn_head
    return heads.MaskrcnnHead(
        params.model_params.architecture.num_classes,
        params.architecture.mask_target_size,
        head_params.num_convs,
        head_params.num_filters,
        head_params.use_separable_conv,
        params.model_params.norm_activation.activation,
        head_params.use_batch_norm,
        norm_activation=norm_activation_generator(params.model_params.norm_activation),
    )


def yolo_v4_head_generator():
    """Generator function for YOLOv4 neck and head architecture"""
    return heads.YOLOv4()
