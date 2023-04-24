"""
 Copyright (c) 2023 Intel Corporation
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
from nncf import ModelType
from nncf import QuantizationPreset
from nncf.common.logging import nncf_logger
from tests.post_training.conftest import MODELS_SCOPE_PATH
from tests.shared.helpers import load_json


def get_preset_enum_by_str(value: str) -> QuantizationPreset:
    """
    Return enum of QuantizationPreset by value.
    """
    for enum_value in QuantizationPreset:
        if enum_value.value == value:
            return enum_value
    raise ValueError(f"QuantizationPreset does not contain {value}")


def get_model_type_enum_by_str(value: str) -> ModelType:
    """
    Return enum of ModelType by value.
    """
    for enum_value in ModelType:
        if enum_value.value == value:
            return enum_value
    raise ValueError(f"ModelType does not contain {value}")


def get_validation_scope() -> dict:
    """
    Read json file that collected models to validation from MODELS_SCOPE_PATH.
    Convert parameters

    :return dict: _description_
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
            qparams["preset"] = get_preset_enum_by_str(qparams["preset"])
        if "model_type" in qparams.keys():
            qparams["model_type"] = get_model_type_enum_by_str(qparams["model_type"])

    return model_scope


VALIDATION_SCOPE = get_validation_scope()
