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
from typing import Callable, Dict, List, Tuple

from torch import nn

from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.torch.graph.node_utils import extraction_potential_fused_modules
from nncf.torch.graph.node_utils import update_fused_bias
from nncf.torch.graph.transformations.commands import PTBiasCorrectionCommand
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTModelExtractionWithFusedBiasCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.nncf_network import PTInsertionPoint


class PTModelTransformer(ModelTransformer):
    """
    Applies transformations upon PyTorch model.
    """

    def __init__(self, model: NNCFNetwork):
        super().__init__(model)
        self._node_to_op_address_mapping = model.get_node_to_op_address_mapping()

    def transform(self, transformation_layout: PTTransformationLayout) -> NNCFNetwork:
        transformations = transformation_layout.transformations

        bias_correction_transformations = []
        extraction_transformations = None
        insertion_transformations = []

        for transformation in transformations:
            if isinstance(transformation, PTInsertionCommand):
                insertion_transformations.append(transformation)
            if isinstance(transformation, PTModelExtractionWithFusedBiasCommand):
                extraction_transformations = transformation
            if isinstance(transformation, PTBiasCorrectionCommand):
                bias_correction_transformations.append(transformation)

        if extraction_transformations:
            return self._apply_extraction_transformations(extraction_transformations)

        if insertion_transformations:
            self._apply_insertion_transformations(insertion_transformations)
        if bias_correction_transformations:
            self._apply_bias_correction_transformations(bias_correction_transformations)
        return self._model

    def _apply_insertion_transformations(self, transformations: List[PTInsertionCommand]) -> None:
        fns_grouped_by_points = {}  # type: Dict[PTInsertionPoint, List[Tuple[Callable, TransformationPriority]]]
        for transformation_command in transformations:  # type: PTInsertionCommand
            target_point = transformation_command.target_point  # type: PTTargetPoint
            target_node_name = target_point.target_node_name
            pt_ip = PTInsertionPoint(
                target_type=target_point.target_type,
                op_address=self._node_to_op_address_mapping[target_node_name],
                input_port_id=target_point.input_port_id,
            )
            fn = transformation_command.fn
            if target_point.type is TargetType.OPERATION_WITH_WEIGHTS:
                fn = UpdateWeight(fn)
            tup = (fn, transformation_command.priority)
            if pt_ip not in fns_grouped_by_points:
                fns_grouped_by_points[pt_ip] = [tup]
            else:
                fns_grouped_by_points[pt_ip].append(tup)

        for pt_ip, fn_list_with_priority in fns_grouped_by_points.items():
            fn_list_with_priority = sorted(fn_list_with_priority, key=lambda x: x[1])
            self._model.insert_at_point(pt_ip, [x[0] for x in fn_list_with_priority])

    def _apply_extraction_transformations(self, transformation: PTModelExtractionWithFusedBiasCommand) -> nn.Module:
        return extraction_potential_fused_modules(transformation.node_name, self._model)

    def _apply_bias_correction_transformations(self, transformations: List[PTBiasCorrectionCommand]) -> None:
        for transformation in transformations:
            update_fused_bias(
                target_node_name=transformation.target_point.target_node_name,
                new_bias=transformation.bias_value,
                model=self._model
            )
