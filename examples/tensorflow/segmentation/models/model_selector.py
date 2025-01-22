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

from examples.tensorflow.segmentation.models.maskrcnn_config import MASKRCNN_CONFIG
from examples.tensorflow.segmentation.models.maskrcnn_model import MaskrcnnModel


def get_predefined_config(model_name):
    if model_name == "MaskRCNN":
        predefined_config = MASKRCNN_CONFIG
    else:
        raise ValueError("Model {} is not supported.".format(model_name))

    return copy.deepcopy(predefined_config)


def get_model_builder(config):
    model_name = config.model

    if model_name == "MaskRCNN":
        model_builder = MaskrcnnModel(config)
    else:
        raise ValueError("Model {} is not supported.".format(model_name))

    return model_builder
