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

from examples.tensorflow.object_detection.preprocessing.retinanet_preprocessing import RetinaNetPreprocessor
from examples.tensorflow.object_detection.preprocessing.yolo_v4_preprocessing import YOLOv4Preprocessor
from examples.tensorflow.segmentation.preprocessing.maskrcnn_preprocessing import MaskRCNNPreprocessor


def get_preprocess_input_fn(config, is_train):
    model_name = config.model
    if model_name == "RetinaNet":
        tfds_decoder, preprocess_input_fn = RetinaNetPreprocessor(config, is_train).create_preprocess_input_fn()
    elif model_name == "MaskRCNN":
        tfds_decoder, preprocess_input_fn = MaskRCNNPreprocessor(config, is_train).create_preprocess_input_fn()
    elif model_name == "YOLOv4":
        tfds_decoder, preprocess_input_fn = YOLOv4Preprocessor(config, is_train).create_preprocess_input_fn()
    else:
        raise ValueError("Unknown model name {}".format(model_name))

    return tfds_decoder, preprocess_input_fn
