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
                      main_name: Optional[str] = None) -> Tuple[List[int], List[str]]:
    """
    Infer model's input_shape and input_name.
    - If both shape and name are given, just return them.
    - If either shape or name is given, find ungiven one using given one.
    - If both shape and name are not given, raise ValueError.
    If the model has dynamic input_shape, both main_shape and main_name must be given.

    :param model: The target model to infer input_shape and input_name.
    :param main_shape: If main_shape is given and equal to one of the model,
        it can be used as it is. If main_name is not given, main_shape is used to find main_name.
    :param main_name: If main_name is given and equal to one of the model,
        it can be used as it is. If main_shape is not given, main_name is used to find main_shape.
    """
    assert len(model.graph.input) > 0
    if main_shape and main_name:
        return main_shape, main_name

    def set_input_shape(node):
        dim = node.type.tensor_type.shape.dim
        shape = []
        for d in dim:
            if 'dimParam' in MessageToDict(d):
                raise ValueError(
                    ('For models with dynamic input_shape, '
                     'input_shape and input_name must be set.'))

            shape.append(int(MessageToDict(d).get("dimValue")))

        return shape

    input_shape = None
    input_name = None
    if main_name:
        nncf_logger.info(
            "input_shape is None. Infer input_shape from the model.")

        for _input in model.graph.input:
            if main_name == _input.name:
                input_shape = set_input_shape(_input)
                input_name = main_name
                break

    elif main_shape:
        nncf_logger.info(
            "input_name is None. Infer input_name from the model.")

        for _input in model.graph.input:
            _input_shape = set_input_shape(_input)
            if main_shape == _input_shape:
                input_shape = main_shape
                input_name = _input.name
                break

    else:
        raise ValueError('Either main_shape or main_name must be set correctly.')

    assert input_shape is not None or input_name is not None, \
        'Either main_shape or main_name must be set correctly.'
    assert isinstance(input_shape, (list, tuple)) and len(input_shape) > 0

    return input_shape, input_name
