"""
 Copyright (c) 2022 Intel Corporation
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

from typing import List, Optional, Tuple

from google.protobuf.json_format import MessageToDict
from nncf.common.utils.logger import logger as nncf_logger
from onnx import ModelProto


def infer_input_shape(model: ModelProto,
                      main_shape: Optional[List[int]] = None,
                      main_keys: Optional[str] = None) -> Tuple[List[int], List[str]]:
    assert len(model.graph.input) > 0

    def set_input_shape(node):
        dim = node.type.tensor_type.shape.dim
        return [int(MessageToDict(d).get("dimValue")) for d in dim]

    if main_shape and main_keys:
        return main_shape, [main_keys]

    if main_keys:
        nncf_logger.info(
            "input_shape is None. Infer input_shape from the model.")

        for _input in model.graph.input:
            if main_keys == _input.name:
                input_shape = set_input_shape(_input)
                input_keys = main_keys
                break

    elif main_shape:
        nncf_logger.info(
            "input_keys is None. Infer input_keys from the model.")
            
        for _input in model.graph.input:
            _input_shape = set_input_shape(_input)
            if main_shape == _input_shape:
                input_shape = main_shape
                input_keys = _input.name
                break

    else:
        raise ValueError('Either main_shape or main_keys must be set correctly.')

    assert len(input_shape) == 4 and input_keys is not None

    return input_shape, [input_keys]
