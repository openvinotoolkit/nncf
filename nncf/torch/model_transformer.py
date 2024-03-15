# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter

from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.graph.transformations.commands import PTBiasCorrectionCommand
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTModelExtractionWithFusedBiasCommand
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import PTWeightUpdateCommand
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.model_analyzer import get_potential_fused_node
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.nncf_network import PTInsertionPoint
from nncf.torch.nncf_network import compression_module_type_to_attr_name
from nncf.torch.quantization.external_quantizer import ExternalOpCallHook
from nncf.torch.utils import get_model_device
from nncf.torch.utils import is_multidevice


class PTModelTransformer(ModelTransformer):
    """
    Applies transformations upon PyTorch model.
    """

    def __init__(self, model: NNCFNetwork):
        super().__init__(model)

        self._command_transformation_ordered_pairs = [
            (PTModelExtractionWithFusedBiasCommand, self._apply_extraction_with_fused_bias_transformations),
            (PTInsertionCommand, self._apply_insertion_transformations),
            (PTSharedFnInsertionCommand, self._apply_shared_nodes_insertion),
            (PTBiasCorrectionCommand, self._apply_bias_correction_transformations),
            (PTWeightUpdateCommand, self._apply_weights_update_transformations),
        ]

    def transform(self, transformation_layout: PTTransformationLayout) -> NNCFNetwork:
        transformations = transformation_layout.transformations
        aggregated_transformations = defaultdict(list)
        requires_graph_rebuild = False
        for transformation in transformations:
            aggregated_transformations[transformation.__class__].append(transformation)
            requires_graph_rebuild = requires_graph_rebuild or transformation.requires_graph_rebuild()

        model = self._model
        for transformation_cls, transformation_fn in self._command_transformation_ordered_pairs:
            transformations = aggregated_transformations[transformation_cls]
            if transformations:
                model = transformation_fn(model, transformations)

        if requires_graph_rebuild:
            model.nncf.rebuild_graph()

        return model

    @staticmethod
    def _apply_insertion_transformations(model: NNCFNetwork, transformations: List[PTInsertionCommand]) -> NNCFNetwork:
        """
        Applies insertion transformations to the model.

        :param model: Model to apply transformations.
        :param transformations: List of the bias correction transformations.
        :return: A modified NNCFNetwork.
        """
        node_to_op_address_mapping = model.nncf.get_node_to_op_address_mapping()
        fns_grouped_by_points: Dict[PTInsertionPoint, List[Tuple[Callable, TransformationPriority]]] = defaultdict(list)

        device = None
        if not is_multidevice(model):
            device = get_model_device(model)

        for transformation_command in transformations:
            target_point: PTTargetPoint = transformation_command.target_point
            target_node_name = target_point.target_node_name
            pt_ip = PTInsertionPoint(
                target_type=target_point.target_type,
                op_address=node_to_op_address_mapping[target_node_name],
                input_port_id=target_point.input_port_id,
                replaced_modules=model.nncf.replace_modules,
            )

            fn = transformation_command.fn
            if device is not None and isinstance(fn, torch.nn.Module):
                fn.to(device)

            if model.nncf.replace_modules and target_point.type is TargetType.OPERATION_WITH_WEIGHTS:
                fn = UpdateWeight(fn)
            tup = (fn, transformation_command)
            fns_grouped_by_points[pt_ip].append(tup)

        for pt_ip, fn_list_with_priority in fns_grouped_by_points.items():
            fn_list_with_priority = sorted(fn_list_with_priority, key=lambda x: x[1].priority)
            for fn, command in fn_list_with_priority:
                model.nncf.insert_at_point(pt_ip, fn, command.hooks_group_name)

        return model

    @staticmethod
    def _apply_shared_node_insertion_with_compression_type(
        model: NNCFNetwork,
        transformations: List[PTSharedFnInsertionCommand],
        compression_module_type: ExtraCompressionModuleType,
    ):
        if not model.nncf.is_compression_module_registered(compression_module_type):
            model.nncf.register_compression_module_type(compression_module_type)

        insertion_commands: List[PTInsertionCommand] = []

        for shared_command in transformations:
            model.nncf.add_compression_module(shared_command.op_name, shared_command.fn, compression_module_type)

            for target_point in shared_command.target_points:
                fn = ExternalOpCallHook(
                    compression_module_type_to_attr_name(compression_module_type), shared_command.op_name
                )
                insertion_commands.append(
                    PTInsertionCommand(
                        target_point,
                        fn,
                        priority=shared_command.priority,
                        hooks_group_name=shared_command.hooks_group_name,
                    )
                )

        return PTModelTransformer._apply_insertion_transformations(model, insertion_commands)

    @staticmethod
    def _apply_shared_nodes_insertion(
        model: NNCFNetwork,
        transformations: List[PTSharedFnInsertionCommand],
    ) -> NNCFNetwork:
        compression_type_vs_transformations = defaultdict(list)
        for transformation in transformations:
            compression_type_vs_transformations[transformation.compression_module_type].append(transformation)

        for compression_module_type, transformations in compression_type_vs_transformations.items():
            model = PTModelTransformer._apply_shared_node_insertion_with_compression_type(
                model, transformations, compression_module_type
            )
        return model

    @staticmethod
    def _apply_extraction_with_fused_bias_transformations(
        model: NNCFNetwork, transformations: List[PTModelExtractionWithFusedBiasCommand]
    ) -> nn.Sequential:
        """
        Extracts copy of sub-modules from the original base on node name and potential fused nodes.

        :param model: Model to apply transformations.
        :param transformation: Model extraction transformation.
        :return: Extracted sub-modules.
        """
        transformation = transformations[-1]
        return extraction_potential_fused_modules(transformation.node_name, model)

    @staticmethod
    def _apply_bias_correction_transformations(
        model: NNCFNetwork, transformations: List[PTBiasCorrectionCommand]
    ) -> NNCFNetwork:
        """
        Applies bias correction transformations on the model.

        :param model: Model to apply transformations.
        :param transformations: List of the bias correction transformations.
        :return: Model with corrected bias.
        """
        for transformation in transformations:
            update_fused_bias(
                target_node_name=transformation.target_point.target_node_name,
                new_bias=transformation.bias_value,
                model=model,
            )
        return model

    @staticmethod
    def _apply_weights_update_transformations(
        model: NNCFNetwork, transformations: List[PTWeightUpdateCommand]
    ) -> NNCFNetwork:
        """
        Applies weight update transformations on the model.

        :param model: Model to apply transformations.
        :param transformations: List of the weight update transformations.
        :return: Model with updated weights.
        """
        for transformation in transformations:
            update_parameter(transformation.target_point.target_node_name, "weight", transformation.weight_value, model)
        return model


def update_fused_bias(target_node_name: str, new_bias: Tensor, model: NNCFNetwork) -> None:
    """
    Update bias for target module or potential fused module.

    :param target_node_name: The target node name.
    :param new_bias: New bias value.
    :param model: The model.
    """
    nncf_graph = model.nncf.get_graph()
    fused_node = get_potential_fused_node(target_node_name, nncf_graph)
    if fused_node is None:
        update_parameter(target_node_name, "bias", new_bias, model)
        return
    target_module = model.nncf.get_containing_module(target_node_name)
    fused_module = model.nncf.get_containing_module(fused_node.node_name)

    if target_module.bias is None:
        update_parameter(fused_node.node_name, "bias", new_bias, model)
        return
    new_bias = new_bias - target_module.bias * fused_module.weight
    update_parameter(fused_node.node_name, "bias", new_bias, model)


def update_parameter(target_node_name: str, parameter_name: str, new_value: Tensor, model: NNCFNetwork) -> None:
    """
    Update parameter for target module.

    :param target_node_name: The target node name.
    :param parameter_name: The name of the parameter to update.
    :param new_value: New parameter value.
    :param model: The model.
    """
    module = model.nncf.get_containing_module(target_node_name)
    parameter: Parameter = getattr(module, parameter_name)
    parameter.data = new_value


def extraction_potential_fused_modules(node_name: str, model: NNCFNetwork) -> nn.Sequential:
    """
    Return Sequential from the copy of module by node_name and potential fused node if exists.

    :param node_name: The node name.
    :param model: The model.
    :return nn.Sequential: Copy of the modules.
    """
    extracted_node_names = [node_name]
    nncf_graph = model.nncf.get_graph()
    fused_node = get_potential_fused_node(node_name, nncf_graph)
    if fused_node:
        extracted_node_names.append(fused_node.node_name)

    extracted_modules = [
        copy.deepcopy(model.nncf.get_containing_module(node_name)) for node_name in extracted_node_names
    ]
    return nn.Sequential(*extracted_modules)
