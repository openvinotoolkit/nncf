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

from typing import List

import numpy as np

from nncf.common.graph.transformations.commands import Command
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationType
from nncf.openvino.graph.node_utils import InplaceInsertionFnType
from nncf.quantization.fake_quantize import FakeQuantizeParameters


class OVTargetPoint(TargetPoint):
    def __init__(self, target_type: TargetType, target_node_name: str, port_id: int):
        super().__init__(target_type)
        self.target_node_name = target_node_name
        self.port_id = port_id

    def __eq__(self, other: "OVTargetPoint") -> bool:
        return (
            isinstance(other, OVTargetPoint)
            and self.type == other.type
            and self.target_node_name == other.target_node_name
            and self.port_id == other.port_id
        )

    def __hash__(self) -> int:
        return hash((self.target_node_name, self.port_id, self._target_type))


class OVInsertionCommand(TransformationCommand):
    def __init__(self, target_point: OVTargetPoint):
        super().__init__(TransformationType.INSERT, target_point)

    def union(self, other: "TransformationCommand") -> "TransformationCommand":
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()


class OVOutputInsertionCommand(OVInsertionCommand):
    def union(self, other: "TransformationCommand") -> "TransformationCommand":
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()


class OVInplaceFnInsertionCommand(OVInsertionCommand):
    def __init__(self, target_point: OVTargetPoint, inplace_op_fn: InplaceInsertionFnType, fn_output_port_id: int):
        super().__init__(target_point)
        self.inplace_op_fn = inplace_op_fn
        self.fn_output_port_id = fn_output_port_id

    def union(self, other: "TransformationCommand") -> "TransformationCommand":
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()


class OVFQNodeRemovingCommand(TransformationCommand):
    """
    Removes FakeQuantize nodes from the model.
    """

    def __init__(self, target_point: OVTargetPoint):
        """
        :param target_point: The TargetPoint instance for the layer that contains information for removing.
        """
        super().__init__(TransformationType.REMOVE, target_point)

    def union(self, other: "TransformationCommand") -> "TransformationCommand":
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()


class OVQuantizerInsertionCommand(OVInsertionCommand):
    def __init__(self, target_point: OVTargetPoint, quantizer_parameters: FakeQuantizeParameters):
        super().__init__(target_point)
        self.quantizer_parameters = quantizer_parameters

    def union(self, other: "TransformationCommand") -> "TransformationCommand":
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()


class OVBiasCorrectionCommand(TransformationCommand):
    """
    Corrects bias value in the model based on the input value.
    """

    def __init__(self, target_point: OVTargetPoint, bias_value: np.ndarray):
        """
        :param target_point: The TargetPoint instance for the correction that contains layer's information.
        :param bias_value: The bias shift value (numpy format) that will be added to the original bias value.
        """
        super().__init__(TransformationType.CHANGE, target_point)
        self.bias_value = bias_value

    def union(self, other: "TransformationCommand") -> "TransformationCommand":
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()


class OVWeightUpdateCommand(TransformationCommand):
    """
    Updates weight value in the model.
    """

    def __init__(self, target_point: OVTargetPoint, weight_value: np.ndarray):
        """
        :param target_point: Target point.
        :param weight_value: New weight value.
        """
        super().__init__(TransformationType.CHANGE, target_point)
        self.weight_value = weight_value

    def union(self, other: "TransformationCommand") -> "TransformationCommand":
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()


class OVModelExtractionCommand(Command):
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


class OVNullBiasInsertionCommand(TransformationCommand):
    """
    Inserts null bias for the corresponding node.
    """

    def __init__(self, target_point: OVTargetPoint):
        """
        :param target_point: The TargetPoint instance for the insertion that contains layer's information.
        """
        super().__init__(TransformationType.INSERT, target_point)

    def union(self, other: "TransformationCommand") -> "TransformationCommand":
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()


class OVMultiplyInsertionCommand(OVInsertionCommand):
    """
    Inserts Multiply nodes before the corresponding nodes.
    """

    def __init__(self, target_point: OVTargetPoint, scale_value: np.ndarray, destination_node_names: List[str]):
        """
        :param target_point: The TargetPoint instance for the insertion that contains layer's information.
        """
        super().__init__(target_point)
        self.scale_value = scale_value
        self.destination_node_names = destination_node_names

    def union(self, other: "TransformationCommand") -> "TransformationCommand":
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()
