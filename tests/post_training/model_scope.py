# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict

from nncf import ModelType
from nncf import QuantizationPreset
from tests.post_training.conftest import MODELS_SCOPE_PATH
from tests.shared.helpers import load_json


def get_validation_scope() -> Dict[str, Any]:
    """
    Read json file that collected models to validation from MODELS_SCOPE_PATH.
    Convert parameters

    :return dict: Dict with model attributes.
    """
    model_scope = load_json(MODELS_SCOPE_PATH)

    for model_name in list(model_scope.keys()):
        model_info = model_scope[model_name]
        if model_info.get("skipped"):
            print(f"Skip {model_name} by '{model_info.get('skipped')}'")
            model_scope.pop(model_name)
            continue

        qparams = model_info["quantization_params"]
        if "preset" in qparams.keys():
            qparams["preset"] = QuantizationPreset[qparams["preset"]]
        if "model_type" in qparams.keys():
            qparams["model_type"] = ModelType[qparams["model_type"]]

    return model_scope


VALIDATION_SCOPE = get_validation_scope()


def get_cached_metric(report_model_name, metric_name):
    cached_metric = None
    try:
        cached_metric = VALIDATION_SCOPE[report_model_name]["metrics"][metric_name]
    except KeyError:
        pass
    return cached_metric
