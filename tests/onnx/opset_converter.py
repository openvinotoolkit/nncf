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

import onnx
from onnx.version_converter import ConvertError
from onnx.version_converter import convert_version

from nncf.common.logging import nncf_logger

TARGET_OPSET_VERSION = 13
TARGET_IR_VERSION = 7


def convert_opset_version(model: onnx.ModelProto, opset_version: int = TARGET_OPSET_VERSION) -> onnx.ModelProto:
    """
    Tries to convert 'model' Opset Version to 'opset_version'.
    If the 'model' can not be converted returns the original 'model'.

    :param model: ONNX model to convert.
    :param opset_version: target Opset Version.
    :return: Converted ONNX model or Original ONNX model.
    """

    try:
        modified_model = convert_version(model, opset_version)
        onnx.checker.check_model(modified_model)
        nncf_logger.info(
            f"The model was successfully converted to the opset version = {modified_model.opset_import[0].version}"
        )
        return modified_model
    except (RuntimeError, ConvertError):
        nncf_logger.error(
            f"Couldn't convert target model to the opset version {opset_version}. Using the copy of the original model"
        )
        return model
