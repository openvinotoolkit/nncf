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
from typing import TypeVar
from typing import Tuple

from nncf.common.factory import ModelTransformerFactory
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFGraph
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.quantization.quantizer_removal import find_quantizer_nodes_to_cut


TModel = TypeVar('TModel')


# TODO(andrey-churkin): We need to introduce common metatypes and
# commands to simplify the method's signature.

def remove_quantizer_from_model(quantized_model: TModel,
                                quantizer_node: NNCFNode,
                                graph: NNCFGraph,
                                quantizer_metatypes: List[OperatorMetatype],
                                const_metatypes: List[OperatorMetatype],
                                quantizable_metatypes: List[OperatorMetatype],
                                quantize_agnostic_metatypes: List[OperatorMetatype],
                                create_command_to_remove_quantizer,
                                create_command_to_update_bias) -> Tuple[TModel, List[NNCFNode], List[NNCFNode]]:
    """
    Finds quantizer nodes that should be removed in addition to `quantizer_node` to get
    the correct model for inference. Removes these quantizers from the `quantized_model`.

    :param quantized_model:
    :param quantizer_node: The quantizer which should be removed.
    :param graph: The graph which was built for `quantized_model`.
    :param quantizer_metatypes: List of quantizer metatypes.
    :param const_metatypes: List of constant metatypes.
    :param quantizable_metatypes: List of metatypes for operations that may be quantized.
    :param quantize_agnostic_metatypes: List of quantize agnostic metatypes.
    :param create_command_to_remove_quantizer: Function to create command to remove quantizer.
    :param create_command_to_update_bias: Function to create command to update bias value.
    :return: A tuple (transformed_model, quantizer_nodes, nodes) where
        - `transformed_model` is the quantized model from which quantizers were removed.
        - `quantizer_nodes` are the quantizers that were removed.
        - `nodes` are the list of nodes that were reverted to their original precision.
    """
    quantizer_nodes, nodes = find_quantizer_nodes_to_cut(graph, quantizer_node,
                                                         quantizer_metatypes, const_metatypes,
                                                         quantizable_metatypes, quantize_agnostic_metatypes)
    transformation_layout = TransformationLayout()

    for node in quantizer_nodes:
        transformation_layout.register(
            create_command_to_remove_quantizer(node)
        )

    for node in nodes:
        original_bias = node.data.get('original_bias', None)
        if original_bias is not None:
            transformation_layout.register(
                create_command_to_update_bias(node, original_bias, graph)
            )

    model_transformer = ModelTransformerFactory.create(quantized_model)
    transformed_model = model_transformer.transform(transformation_layout)

    return transformed_model, quantizer_nodes, nodes
