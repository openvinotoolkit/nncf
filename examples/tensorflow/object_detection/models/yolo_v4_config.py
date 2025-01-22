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

YOLOV4_CONFIG = SampleConfig(
    {
        "preprocessing": {
            "enhance_mosaic_augment": True,
            "multi_anchor_assign": True,
        },
        "iou_threshold": 0.5,
        "input_shape": (608, 608),
        "anchors": [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
        "elim_grid_sense": True,
        "model_params": {
            "architecture": {
                "backbone": {"name": "darknet"},
            },
            "num_classes": 80,
            "num_feature_layers": 3,
            "loss_params": {
                "ignore_thresh": 0.5,
                "label_smoothing": 0,
                "use_focal_loss": False,
                "use_focal_obj_loss": False,
                "use_softmax_loss": False,
                "use_giou_loss": False,
                "use_diou_loss": True,
            },
        },
        "postprocessing": {
            "conf_threshold": 0.001,
        },
    }
)
