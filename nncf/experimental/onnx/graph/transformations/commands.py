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

from typing import Optional, List
import numpy as np

from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationType
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.experimental.onnx.algorithms.quantization.utils import QuantizerLayerParameters


class ONNXTargetPoint(TargetPoint):
    def __init__(self, target_type: TargetType, target_node_name: str, edge_name: Optional[str] = None):
        super().__init__(target_type)
        self.target_node_name = target_node_name
        self.edge_name = edge_name

    def __eq__(self, other: 'ONNXTargetPoint') -> bool:
        return isinstance(other, ONNXTargetPoint) and \
               self.type == other.type and self.target_node_name == other.target_node_name and \
               self.edge_name == other.edge_name

    def __hash__(self) -> int:
        return hash((self.target_node_name, self.edge_name, self._target_type))

    def __lt__(self, other: 'ONNXTargetPoint') -> bool:
        # The ONNXTargetPoint should have the way to compare.
        # NNCF has to be able returning the Quantization Target Points in the deterministic way.
        # MinMaxQuantizationAlgorithm returns the sorted Set of such ONNXTargetPoints.
        params = ['_target_type', 'target_node_name', 'edge_name']
        for param in params:
            if self.__getattribute__(param) < other.__getattribute__(param):
                return True
            if self.__getattribute__(param) > other.__getattribute__(param):
                return False
        return False


class ONNXInsertionCommand(TransformationCommand):
    def __init__(self, target_point: ONNXTargetPoint):
        super().__init__(TransformationType.INSERT, target_point)

    def union(self, other: 'TransformationCommand') -> 'TransformationCommand':
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()


class ONNXQuantizerInsertionCommand(ONNXInsertionCommand):
    def __init__(self, target_point: ONNXTargetPoint, quantizer_parameters: QuantizerLayerParameters):
        super().__init__(target_point)
        self.quantizer_parameters = quantizer_parameters

    def union(self, other: 'TransformationCommand') -> 'TransformationCommand':
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()


class ONNXOutputInsertionCommand(ONNXInsertionCommand):
    def union(self, other: 'TransformationCommand') -> 'TransformationCommand':
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()

class ONNXBiasCorrectionCommand(TransformationCommand):
    def __init__(self, target_point: ONNXTargetPoint, bias_value: np.array):
        super().__init__(TransformationType.CHANGE, target_point)
        self.bias_value = bias_value

    def union(self, other: 'TransformationCommand') -> 'TransformationCommand':
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()
