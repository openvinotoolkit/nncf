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
import openvino.runtime as ov

from nncf.common.factory import ModelTransformerFactory
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionBackpropDataMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVDepthwiseConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVGroupConvolutionBackpropDataMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVGroupConvolutionMetatype
from nncf.openvino.graph.node_utils import is_node_with_bias
from nncf.openvino.graph.transformations.command_creation import OVCommandCreator


def insert_null_biases(model: ov.Model) -> ov.Model:
    """
    This method finds and inserts zero biases for the layers that should have it.

    :param model: ov.Model instance.
    :return: Updated ov.Model instance with zero biases
    """
    types_to_insert_bias = [
        OVConvolutionMetatype,
        OVGroupConvolutionMetatype,
        OVDepthwiseConvolutionMetatype,
        OVConvolutionBackpropDataMetatype,
        OVGroupConvolutionBackpropDataMetatype,
    ]
    nncf_graph = NNCFGraphFactory.create(model)
    nodes_without_biases = nncf_graph.get_nodes_by_metatypes(types_to_insert_bias)
    nodes_without_biases = [node for node in nodes_without_biases if not is_node_with_bias(node, nncf_graph)]
    transformation_layout = TransformationLayout()
    model_transformer = ModelTransformerFactory.create(model)
    for node_without_bias in nodes_without_biases:
        bias_insertion_command = OVCommandCreator.create_command_to_insert_bias(node_without_bias)
        transformation_layout.register(bias_insertion_command)
    return model_transformer.transform(transformation_layout)
