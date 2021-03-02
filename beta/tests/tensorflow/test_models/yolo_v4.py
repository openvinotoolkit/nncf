"""
 Copyright (c) 2021 Intel Corporation
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

from beta.examples.tensorflow.common.sample_config import SampleConfig
from beta.examples.tensorflow.object_detection.models.model_selector import get_model_builder
from beta.examples.tensorflow.object_detection.models.model_selector import get_predefined_config
from beta.tests.conftest import TEST_ROOT


def YOLOv4(input_shape=None):
    NNCF_ROOT = TEST_ROOT.parent
    path_to_config = NNCF_ROOT.joinpath('examples', 'tensorflow', 'object_detection', 'configs', 'yolo_v4_coco.json')

    config_from_json = SampleConfig.from_json(path_to_config)
    predefined_config = get_predefined_config(config_from_json.model)
    predefined_config.update(config_from_json)

    model_builder = get_model_builder(predefined_config)
    model = model_builder.build_model()

    return model
