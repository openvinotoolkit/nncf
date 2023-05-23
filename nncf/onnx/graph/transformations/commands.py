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

from typing import Dict, List, Optional, Tuple

import numpy as np

from nncf.common.graph.transformations.commands import Command
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationType
from nncf.onnx.quantization.quantizer_parameters import ONNXQuantizerLayerParameters


class ONNXTargetPoint(TargetPoint):
    def __init__(self, target_type: TargetType, target_node_name: str, port_id: Optional[int] = None):
        """
        Constructor.

        :param target_type: Target type of the target point.
        :param target_node_name: ONNX node name.
        :param port_id: Number of port id in case target_type is
            TargetType.PRE_LAYER_OPERATION or TargetType.POST_LAYER_OPERATION.
        """
        super().__init__(target_type)
        self.target_node_name = target_node_name
        self.port_id = port_id

    def __eq__(self, other: "ONNXTargetPoint") -> bool:
        return (
            isinstance(other, ONNXTargetPoint)
            and self.type == other.type
            and self.target_node_name == other.target_node_name
            and self.port_id == other.port_id
        )

    def __hash__(self) -> int:
        return hash((self.target_node_name, self.port_id, self._target_type))


class ONNXInsertionCommand(TransformationCommand):
    def __init__(self, target_point: ONNXTargetPoint, input_edges_mapping: Dict[str, Tuple[str, int]]):
        super().__init__(TransformationType.INSERT, target_point)
        # Insertion command could be applied to NNCF input nodes, e.g.
        # quantizers will be tied with POST OP of NNCF input nodes.
        # To get the ONNX edge to apply a command,
        # need to keep the mapping NNCF input nodes to the following ONNX nodes.
        self.input_edges_mapping = input_edges_mapping

    def union(self, other: "TransformationCommand") -> "TransformationCommand":
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()


class ONNXQuantizerInsertionCommand(ONNXInsertionCommand):
    def __init__(
        self,
        target_point: ONNXTargetPoint,
        nncf_input_node_next_onnx_nodes: Dict[str, List[str]],
        quantizer_parameters: ONNXQuantizerLayerParameters,
    ):
        super().__init__(target_point, nncf_input_node_next_onnx_nodes)
        self.quantizer_parameters = quantizer_parameters

    def union(self, other: "TransformationCommand") -> "TransformationCommand":
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()


class ONNXOutputInsertionCommand(ONNXInsertionCommand):
    def union(self, other: "TransformationCommand") -> "TransformationCommand":
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

    def union(self, other: "TransformationCommand") -> "TransformationCommand":
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()


class ONNXModelExtractionCommand(Command):
    """
    Extracts sub-graph based on the sub-model input and output names.
    """

    def __init__(self, inputs: List[str], outputs: List[str]):
        """
        :param inputs: List of the input names that denote the sub-graph beginning.
        :param outputs: List of the output names that denote the sub-graph ending.
        """
        super().__init__(TransformationType.EXTRACT)
        self.inputs = inputs
        self.outputs = outputs

    def union(self, other: "Command") -> "Command":
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

    def union(self, other: "TransformationCommand") -> "TransformationCommand":
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()


class ONNXNullBiasInsertionCommand(TransformationCommand):
    """
    Inserts null bias for the corresponding node.
    """

    def __init__(self, target_point: ONNXTargetPoint):
        """
        :param target_point: The TargetPoint instance for the insertion that contains layer's information.
        """
        super().__init__(TransformationType.INSERT, target_point)

    def union(self, other: "TransformationCommand") -> "TransformationCommand":
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()
