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

from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.patterns import GraphPattern

from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXSigmoidMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXHardSigmoidMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXAddLayerMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXSubMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXMulLayerMetatype


def _create_elememntwise_op_without_const(pattern, elementwise_metatype):
    if elementwise_metatype == ONNXAddLayerMetatype:
        label_attr = 'ADD'
    if elementwise_metatype == ONNXMulLayerMetatype:
        label_attr = 'MUL'
    if elementwise_metatype == ONNXSubMetatype:
        label_attr = 'SUB'
    elementwise_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: label_attr,
                                           GraphPattern.METATYPE_ATTR: elementwise_metatype})
    return elementwise_node


def create_swish_activation() -> GraphPattern:
    pattern = GraphPattern()

    input_pattern_node_1 = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: '*INPUT_NODE*',
           GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE})
    sigmoid_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SIGMOID',
                                         GraphPattern.METATYPE_ATTR: ONNXSigmoidMetatype})
    mul_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MUL',
                                     GraphPattern.METATYPE_ATTR: ONNXMulLayerMetatype})

    pattern.add_edge(input_pattern_node_1, sigmoid_node_1)
    pattern.add_edge(input_pattern_node_1, mul_node_1)
    pattern.add_edge(sigmoid_node_1, mul_node_1)

    input_pattern_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: '*INPUT_NODE*',
                                               GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE})
    sigmoid_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'HARDSIGMOID',
                                         GraphPattern.METATYPE_ATTR: ONNXHardSigmoidMetatype})
    mul_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MUL',
                                     GraphPattern.METATYPE_ATTR: ONNXMulLayerMetatype})

    pattern.add_edge(input_pattern_node_2, sigmoid_node_2)
    pattern.add_edge(input_pattern_node_2, mul_node_2)
    pattern.add_edge(sigmoid_node_2, mul_node_2)

    return pattern


def create_input_preprocessing_pattern() -> GraphPattern:
    pattern = GraphPattern()
    el_nodes_metatypes = [(ONNXAddLayerMetatype, ONNXMulLayerMetatype), (ONNXMulLayerMetatype, ONNXAddLayerMetatype)]
    for el_nodes_metatype in el_nodes_metatypes:
        model_input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                               GraphPattern.METATYPE_ATTR: NNCFGraphNodeType.INPUT_NODE})
        first_el_node = _create_elememntwise_op_without_const(pattern, el_nodes_metatype[0])
        second_el_node = _create_elememntwise_op_without_const(pattern, el_nodes_metatype[1])

        pattern.add_edge(model_input_node, first_el_node)
        pattern.add_edge(first_el_node, second_el_node)

    el_nodes_metatypes = [ONNXAddLayerMetatype, ONNXMulLayerMetatype]
    for el_nodes_metatype in el_nodes_metatypes:
        model_input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                               GraphPattern.METATYPE_ATTR: NNCFGraphNodeType.INPUT_NODE})
        el_node = _create_elememntwise_op_without_const(pattern, el_nodes_metatype)

        pattern.add_edge(model_input_node, el_node)
    return pattern


def create_scale_shift() -> GraphPattern:
    pattern = GraphPattern()
    el_nodes_metatypes = [(ONNXAddLayerMetatype, ONNXMulLayerMetatype), (ONNXMulLayerMetatype, ONNXAddLayerMetatype)]
    for el_nodes_metatype in el_nodes_metatypes:
        model_input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: '*INPUT_NODE*',
                                               GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE})
        first_el_node = _create_elememntwise_op_without_const(pattern, el_nodes_metatype[0])
        second_el_node = _create_elememntwise_op_without_const(pattern, el_nodes_metatype[1])

        pattern.add_edge(model_input_node, first_el_node)
        pattern.add_edge(first_el_node, second_el_node)

    return pattern
