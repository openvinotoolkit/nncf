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

from tensorflow.keras import applications as keras_models

from examples.tensorflow.common import models as custom_models

AVAILABLE_MODELS = dict(keras_models.__dict__)
AVAILABLE_MODELS.update(custom_models.__dict__)


def get_model(model_name, input_shape=None, pretrained=True, num_classes=1000, weights=None):
    if model_name in AVAILABLE_MODELS:
        model = AVAILABLE_MODELS[model_name]
    else:
        raise Exception("Undefined model name: {}".format(model_name))

    model_params = {"classes": num_classes}
    if weights is not None:
        model_params["weights"] = weights
    elif not pretrained:
        model_params["weights"] = None
    if input_shape is not None:
        model_params["input_shape"] = tuple(input_shape[1:])

    return model, model_params
