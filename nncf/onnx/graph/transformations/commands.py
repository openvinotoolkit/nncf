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

from typing import List
import numpy as np

from nncf.common.graph.transformations.commands import Command, TransformationCommand
from nncf.common.graph.transformations.commands import TransformationType
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.onnx.quantization.quantizer_parameters import ONNXQuantizerLayerParameters


class ONNXTargetPoint(TargetPoint):
    def __init__(self, target_type: TargetType, target_node_name: str, port_id: int):
        super().__init__(target_type)
        self.target_node_name = target_node_name
        self.port_id = port_id

    def __eq__(self, other: 'ONNXTargetPoint') -> bool:
        return isinstance(other, ONNXTargetPoint) and \
               self.type == other.type and self.target_node_name == other.target_node_name and \
               self.port_id == other.port_id

    def __hash__(self) -> int:
        return hash((self.target_node_name, self.port_id, self._target_type))

    def __lt__(self, other: 'ONNXTargetPoint') -> bool:
        # The ONNXTargetPoint should have the way to compare.
        # NNCF has to be able returning the Quantization Target Points in the deterministic way.
        # MinMaxQuantizationAlgorithm returns the sorted Set of such ONNXTargetPoints.
        params = ['_target_type', 'target_node_name', 'port_id']
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
    def __init__(self, target_point: ONNXTargetPoint, quantizer_parameters: ONNXQuantizerLayerParameters):
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
    """
    Corrects bias value in the model based on the input value.
    """
    def __init__(self, target_point: ONNXTargetPoint, bias_value: np.ndarray):
        """
        :param target_point: The TargetPoint instance for the correction that contains layer's information.
        :param bias_value: The bias shift value (numpy format) that will be added to the original bias value.
        """
        super().__init__(TransformationType.CHANGE, target_point)
        self.bias_value = bias_value

    def union(self, other: 'TransformationCommand') -> 'TransformationCommand':
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()

class ONNXModelExtractionCommand(Command):
    """
    Extracts sub-graph based on the sub-model input and output names.
    """
    def __init__(self, inputs: List[str], outputs: List[str]):
        """
        :param inputs: List of the input names that denote the sub-graph beggining.
        :param outputs: List of the output names that denote the sub-graph ending.
        """
        super().__init__(TransformationType.EXTRACT)
        self.inputs = inputs
        self.outputs = outputs

    def union(self, other: 'Command') -> 'Command':
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()

class ONNXQDQNodeRemovingCommand(TransformationCommand):
    """
    Removes Quantizer or Dequantizer nodes from the model.
    """

    def __init__(self, target_point: ONNXTargetPoint):
        """
        :param target_point: The TargetPoint instance for the layer that contains information for removing.
        """
        super().__init__(TransformationType.REMOVE, target_point)

    def union(self, other: 'TransformationCommand') -> 'TransformationCommand':
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()
