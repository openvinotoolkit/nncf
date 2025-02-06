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

import copy

from examples.tensorflow.object_detection.models.retinanet_config import RETINANET_CONFIG
from examples.tensorflow.object_detection.models.retinanet_model import RetinanetModel
from examples.tensorflow.object_detection.models.yolo_v4_config import YOLOV4_CONFIG
from examples.tensorflow.object_detection.models.yolo_v4_model import YOLOv4Model


def get_predefined_config(model_name):
    if model_name == "RetinaNet":
        predefined_config = RETINANET_CONFIG
    elif model_name == "YOLOv4":
        predefined_config = YOLOV4_CONFIG
    else:
        raise ValueError("Model {} is not supported.".format(model_name))

    return copy.deepcopy(predefined_config)


def get_model_builder(config):
    model_name = config.model

    if model_name == "RetinaNet":
        model_builder = RetinanetModel(config)
    elif model_name == "YOLOv4":
        model_builder = YOLOv4Model(config)
    else:
        raise ValueError("Model {} is not supported.".format(model_name))

    return model_builder
