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

from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationType
from nncf.experimental.onnx.algorithms.quantization.utils import QuantizerLayerParameters


class ONNXInsertionCommand(TransformationCommand):
    def __init__(self, target_layer_name: str):
        # TODO (kshpv): align target_layer_name
        super().__init__(TransformationType.INSERT, target_layer_name)

    def union(self, other: 'TransformationCommand') -> 'TransformationCommand':
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()


class ONNXQuantizerInsertionCommand(ONNXInsertionCommand):
    def __init__(self, target_layer_name: str, quantizer_parameters: QuantizerLayerParameters):
        super().__init__(target_layer_name)
        self.quantizer_parameters = quantizer_parameters

    def union(self, other: 'TransformationCommand') -> 'TransformationCommand':
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()
