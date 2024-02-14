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

from typing import Iterable, List, Optional, Tuple, Union

import torch

import nncf
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import CONST_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.tensor.definitions import TensorDataType
from nncf.experimental.tensor.tensor import Tensor
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.weight_lowering import compress_weight
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.graph import operator_metatypes as om
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import WeightsDecompressor
from nncf.torch.tensor_statistics.collectors import get_raw_stat_collector


def split_weight_name(weight_name: str) -> Tuple[str, str]:
    index = weight_name.rfind(".")
    if index == -1:
        return str(), weight_name
    module_name = weight_name[:index]
    weight_attr_name = weight_name[index + 1 :]
    return module_name, weight_attr_name


def get_module_by_name(module_name: str, model: torch.nn.Module) -> torch.nn.Module:
    if not module_name:
        return model
    curr_module = model
    for name in module_name.split("."):
        for child_name, child_module in curr_module.named_children():
            if child_name == name:
                curr_module = child_module
                break
        else:
            raise nncf.ModuleNotFoundError(f"Could not find the {module_name} module in the model.")
    return curr_module


def find_weight_node_in_constant_subgraph(node: NNCFNode, graph: NNCFGraph) -> Union[NNCFNode, None]:
    if node.metatype == om.PTNoopMetatype:
        prev_nodes = graph.get_previous_nodes(node)
        if len(prev_nodes) != 1:
            return None
        return find_weight_node_in_constant_subgraph(prev_nodes[0], graph)
    if node.metatype in CONST_NOOP_METATYPES:
        return node
    return None


def get_weight_node(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> NNCFNode:
    for prev_node in graph.get_previous_nodes(node_with_weight):
        edge = graph.get_edge(prev_node, node_with_weight)
        if edge.input_port_id == weight_port_id:
            weight_node = find_weight_node_in_constant_subgraph(prev_node, graph)
            if weight_node is None:
                raise nncf.InternalError("Could not find a constant node in the model graph.")
            return weight_node


class PTWeightCompressionAlgoBackend(WeightCompressionAlgoBackend):
    TARGET_TYPE_TO_PT_INS_TYPE_MAP = {
        TargetType.PRE_LAYER_OPERATION: TargetType.OPERATOR_PRE_HOOK,
        TargetType.POST_LAYER_OPERATION: TargetType.OPERATOR_POST_HOOK,
    }
    MATMUL_METATYPES = [om.PTLinearMetatype, om.PTMatMulMetatype, om.PTAddmmMetatype]
    EMBEDDING_METATYPES = [om.PTEmbeddingMetatype]
    CONVOLUTION_METATYPES = [
        om.PTConv1dMetatype,
        om.PTConv2dMetatype,
        om.PTConv3dMetatype,
        om.PTDepthwiseConv1dSubtype,
        om.PTDepthwiseConv2dSubtype,
        om.PTDepthwiseConv3dSubtype,
        om.PTConvTranspose1dMetatype,
        om.PTConvTranspose2dMetatype,
        om.PTConvTranspose3dMetatype,
    ]

    @property
    def matmul_metatypes(self) -> List[OperatorMetatype]:
        return PTWeightCompressionAlgoBackend.MATMUL_METATYPES

    @property
    def embedding_metatypes(self) -> List[OperatorMetatype]:
        return PTWeightCompressionAlgoBackend.EMBEDDING_METATYPES

    @property
    def convolution_metatypes(self) -> List[OperatorMetatype]:
        return PTWeightCompressionAlgoBackend.CONVOLUTION_METATYPES

    @staticmethod
    def is_node_with_weights(node: NNCFNode, graph: NNCFGraph) -> bool:
        if (
            node.metatype not in PTWeightCompressionAlgoBackend.MATMUL_METATYPES
            and node.metatype not in PTWeightCompressionAlgoBackend.EMBEDDING_METATYPES
            and node.metatype not in PTWeightCompressionAlgoBackend.CONVOLUTION_METATYPES
        ):
            return False
        for prev_node in graph.get_previous_nodes(node):
            edge = graph.get_edge(prev_node, node)
            if edge.input_port_id not in node.metatype.weight_port_ids:
                continue
            weight_node = find_weight_node_in_constant_subgraph(prev_node, graph)
            if weight_node is not None:
                return True
        return False

    @staticmethod
    def get_weight_names_and_port_ids(node: NNCFNode, graph: NNCFGraph) -> List[Tuple[str, int]]:
        weight_port_ids = []
        for prev_node in graph.get_previous_nodes(node):
            weight_node = find_weight_node_in_constant_subgraph(prev_node, graph)
            if weight_node is None:
                continue
            edge = graph.get_edge(prev_node, node)
            if edge.input_port_id in node.metatype.weight_port_ids:
                weight_port_ids.append((weight_node.layer_attributes.name, edge.input_port_id))
        return weight_port_ids

    @staticmethod
    def get_reduction_axes(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> Optional[Tuple[int]]:
        weight_node = get_weight_node(node_with_weight, weight_port_id, graph)

        ndims = len(weight_node.layer_attributes.shape)
        reduction_axes = None
        if node_with_weight.metatype == om.PTEmbeddingMetatype:
            reduction_axes = [1]
        elif node_with_weight.metatype == om.PTLinearMetatype:
            reduction_axes = [ndims - 1]
        elif node_with_weight.metatype == om.PTMatMulMetatype:
            if weight_port_id == 0:
                reduction_axes = [ndims - 1]
            elif weight_port_id == 1:
                reduction_axes = [max(0, ndims - 2)]
        elif node_with_weight.metatype == om.PTAddmmMetatype:
            if weight_port_id == 1:
                reduction_axes = [ndims - 1]
            elif weight_port_id == 2:
                reduction_axes = [max(0, ndims - 2)]
        elif node_with_weight.metatype in PTWeightCompressionAlgoBackend.CONVOLUTION_METATYPES:
            channel_idx = (
                1
                if node_with_weight.metatype
                in [om.PTConvTranspose1dMetatype, om.PTConvTranspose2dMetatype, om.PTConvTranspose3dMetatype]
                else 0
            )
            reduction_axes = [i for i in range(ndims) if i != channel_idx]
        return tuple(reduction_axes)

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        if NNCFGraphNodeType.INPUT_NODE in target_node_name or target_type == TargetType.POST_LAYER_OPERATION:
            port_id = None
        if target_type in PTWeightCompressionAlgoBackend.TARGET_TYPE_TO_PT_INS_TYPE_MAP:
            target_type = PTWeightCompressionAlgoBackend.TARGET_TYPE_TO_PT_INS_TYPE_MAP[target_type]
        return PTTargetPoint(target_type, target_node_name, input_port_id=port_id)

    @staticmethod
    def raw_statistic_collector(num_samples: Optional[int] = None) -> TensorCollector:
        return get_raw_stat_collector(num_samples)

    @staticmethod
    def get_activation_port_id(node: NNCFNode, graph: NNCFGraph) -> int:
        activation_ports = []
        for prev_node in graph.get_previous_nodes(node):
            if prev_node.metatype in CONST_NOOP_METATYPES:
                continue
            edge = graph.get_edge(prev_node, node)
            activation_ports.append(edge.input_port_id)
        assert len(activation_ports) == 1
        return activation_ports[0]

    def get_weight(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: torch.nn.Module, graph: NNCFGraph
    ) -> Tensor:
        weight_node = get_weight_node(node_with_weight, weight_port_id, graph)
        weight_name = weight_node.layer_attributes.name
        module_name, weight_attr_name = split_weight_name(weight_name)
        module = get_module_by_name(module_name, model)
        weight = getattr(module, weight_attr_name)
        if weight is None or not isinstance(weight, torch.nn.Parameter):
            raise nncf.InternalError(f"Could not find a torch.nn.Parameter in the model by name {weight_name}.")

        return Tensor(weight)

    def set_weight(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: torch.nn.Module, graph: NNCFGraph, weight: Tensor
    ):
        pass

    def transform_model(
        self, model: NNCFNetwork, graph: NNCFGraph, weight_compression_parameters: Iterable[WeightCompressionParameters]
    ) -> NNCFNetwork:
        transformation_layout = TransformationLayout()

        for wc_params in weight_compression_parameters:
            compression_config = wc_params.compression_config
            if compression_config.mode not in [
                CompressWeightsMode.INT8_ASYM,
                CompressWeightsMode.INT8_SYM,
                CompressWeightsMode.INT8,
            ]:
                raise ValueError(f"{compression_config.mode.value} is not supported.")

            weight_node = get_weight_node(wc_params.node_with_weight, wc_params.weight_port_id, graph)
            weight_name = weight_node.layer_attributes.name
            module_name, weight_attr_name = split_weight_name(weight_name)
            module = get_module_by_name(module_name, model)
            weight = getattr(module, weight_attr_name)
            if weight is None or not isinstance(weight, torch.nn.Parameter):
                raise nncf.InternalError(f"Could not find a torch.nn.Parameter in the model by name {weight_name}.")

            # calculates compressed weights and decompression parameters
            compressed_weight = compress_weight(Tensor(weight), wc_params.reduction_axes, compression_config)

            # pack compressed tensor
            packed_tensor = compressed_weight.tensor.astype(TensorDataType.uint8)

            # sets compressed tensor
            compressed_parameter = torch.nn.Parameter(packed_tensor.data, requires_grad=False)
            setattr(module, weight_attr_name, compressed_parameter)

            consumer_nodes = graph.get_next_nodes(weight_node)
            if len(consumer_nodes) > 1:
                for c_node in consumer_nodes:
                    c_module = model.nncf.get_module_by_scope(Scope.from_str(c_node.layer_name))
                    for name, param in c_module.named_parameters(recurse=False, remove_duplicate=False):
                        if id(param) == id(weight):
                            setattr(c_module, name, compressed_parameter)

            # pack zero point tensor
            packed_zero_point = compressed_weight.zero_point.astype(TensorDataType.uint8)

            # creates weight decompressor
            decompressor = WeightsDecompressor(compressed_weight.scale.data, packed_zero_point.data)

            # registry weight decompression module in the model
            decompressor_name = f"weights_decompressor_{weight_node.node_name.replace('.', '_')}"

            # inserts the weight decompressor into the model as the post hook on the model weight
            transformation_layout.register(
                PTSharedFnInsertionCommand(
                    [PTTargetPoint(TargetType.OPERATOR_POST_HOOK, target_node_name=weight_node.node_name)],
                    decompressor,
                    decompressor_name,
                )
            )

        # apply transformations
        transformed_model = PTModelTransformer(model).transform(transformation_layout)

        return transformed_model
