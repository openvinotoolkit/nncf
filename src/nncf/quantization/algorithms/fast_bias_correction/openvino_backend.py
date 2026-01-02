# # Copyright (c) 2025 Intel Corporation
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #      http://www.apache.org/licenses/LICENSE-2.0
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# from copy import deepcopy
# from typing import Optional, Tuple, Dict, List

# import numpy as np
# from openvino.runtime import Model

# from nncf.common.graph import NNCFGraph, NNCFNode
# from nncf.common.graph.transformations.commands import TargetType
# from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
# from nncf.openvino.graph.metatypes.groups import OPERATIONS_WITH_BIAS_REDUCED
# from nncf.openvino.graph.node_utils import (
#     get_act_quantization_axis,
#     get_bias_value,
#     is_any_weight_quantized,
#     is_node_with_bias,
# )
# from nncf.openvino.graph.transformations.command_creation import create_bias_correction_command
# from nncf.openvino.graph.transformations.commands import (
#     OVInitializerUpdateCommand,
#     OVModelExtractionCommand,
#     OVTargetPoint,
# )
# from nncf.openvino.statistics.collectors import get_mean_statistic_collector
# from nncf.quantization.algorithms.fast_bias_correction.backend import FastBiasCorrectionAlgoBackend
# from nncf.tensor import Tensor


# class OVFastBiasCorrectionAlgoBackend(FastBiasCorrectionAlgoBackend):
#     """
#     OpenVINO backend implementation of Fast Bias Correction algorithm.
#     """

#     def __init__(self, model: Model):
#         self._model = model

#     @staticmethod
#     def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
#         return OVTargetPoint(target_type, target_node_name, port_id)

#     @staticmethod
#     def create_bias_correction_command(
#         node: NNCFNode, bias_value: Tensor, nncf_graph: NNCFGraph
#     ) -> OVInitializerUpdateCommand:
#         return create_bias_correction_command(node, bias_value.data)

#     @staticmethod
#     def model_extraction_command(
#         input_ids: List[Tuple[str, int]], output_ids: List[Tuple[str, int]]
#     ) -> OVModelExtractionCommand:
#         return OVModelExtractionCommand(input_ids, output_ids)

#     @staticmethod
#     def mean_statistic_collector(
#         channel_axis: int,
#         inplace: bool,
#         num_samples: Optional[int] = None,
#         window_size: Optional[int] = None,
#     ) -> TensorCollector:
#         return get_mean_statistic_collector(num_samples, channel_axis, window_size, inplace)

#     @staticmethod
#     def get_sub_input_output_names(subgraph: Model) -> Tuple[str, str]:
#         return subgraph.inputs[0].get_any_name(), subgraph.outputs[0].get_any_name()

#     @staticmethod
#     def create_input_data(
#         shape: Tuple[int, ...],
#         data: List[Tensor],
#         input_name: str,
#         channel_axis: int,
#     ) -> Dict[str, np.ndarray]:
#         """
#         Constructs OpenVINO model input tensor by filling per-channel data along the given axis.
#         """
#         blob = np.zeros(shape, dtype=data[0].data.dtype)
#         num_channels = shape[channel_axis]
#         for j in range(num_channels):
#             index = tuple(slice(None) if i != channel_axis else j for i in range(len(shape)))
#             blob[index] = data[j].data
#         return {input_name: blob}

#     @staticmethod
#     def get_bias_value(node: NNCFNode, nncf_graph: NNCFGraph, model: Model) -> Tensor:
#         return Tensor(get_bias_value(node, model))

#     @staticmethod
#     def get_activation_port_ids_for_bias_node(node: NNCFNode) -> Tuple[int, int]:
#         activation_port = 0
#         if getattr(node.metatype, "possible_weight_ports", None):
#             activation_ports = deepcopy(node.metatype.possible_weight_ports)
#             weight_ports = getattr(getattr(node, "layer_attributes", None), "weight_attrs", [])
#             for weight_port in weight_ports:
#                 if weight_port in activation_ports:
#                     activation_ports.remove(weight_port)
#             if len(activation_ports) == 1:
#                 activation_port = activation_ports[0]
#         return activation_port, 0

#     @staticmethod
#     def process_model_output(raw_data: Dict[str, np.ndarray], output_name: str) -> Tensor:
#         return Tensor(raw_data[output_name])

#     @staticmethod
#     def is_quantized_weights(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
#         return is_any_weight_quantized(node, nncf_graph)

#     @staticmethod
#     def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
#         return is_node_with_bias(node) and node.metatype in OPERATIONS_WITH_BIAS_REDUCED

#     @staticmethod
#     def get_node_names_for_input_output_statistics(node: NNCFNode, nncf_graph: NNCFGraph) -> Tuple[str, str]:
#         return node.node_name, node.node_name

#     @staticmethod
#     def get_activation_channel_axis(node: NNCFNode, port_id: int, input_shape: Tuple[int, ...]) -> int:
#         return get_act_quantization_axis(node, port_id)


print("hello")
