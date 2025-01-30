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
from examples.common.sample_config import SampleConfig

RETINANET_CONFIG = SampleConfig(
    {
        "preprocessing": {
            "match_threshold": 0.5,
            "unmatched_threshold": 0.5,
            "aug_rand_hflip": True,
            "aug_scale_min": 1.0,
            "aug_scale_max": 1.0,
            "skip_crowd_during_training": True,
            "max_num_instances": 100,
        },
        "model_params": {
            "architecture": {
                "backbone": {"name": "resnet", "params": {"depth": 50}},
                "min_level": 3,
                "max_level": 7,
                "multilevel_features": "fpn",
                "fpn_params": {"fpn_feat_dims": 256, "use_separable_conv": False, "use_batch_norm": True},
                "num_classes": 91,
                "head_params": {"num_convs": 4, "num_filters": 256, "use_separable_conv": False},
            },
            "anchor": {"num_scales": 3, "aspect_ratios": [1.0, 2.0, 0.5], "anchor_size": 4.0},
            "loss_params": {
                "focal_loss_alpha": 0.25,
                "focal_loss_gamma": 1.5,
                "huber_loss_delta": 0.1,
                "box_loss_weight": 50,
            },
            "norm_activation": {
                "activation": "relu",
                "batch_norm_momentum": 0.997,
                "batch_norm_epsilon": 0.0001,
                "use_sync_bn": False,
            },
            "postprocessing": {
                "use_batched_nms": False,
                "max_total_size": 100,
                "nms_iou_threshold": 0.5,
                "score_threshold": 0.05,
                "pre_nms_num_boxes": 5000,
            },
        },
    }
)
