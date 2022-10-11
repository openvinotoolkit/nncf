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

from typing import List

import pytest

from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNX_OPERATION_METATYPES
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.quantization.quantizer_propagation.solver import PostprocessingNodeLocator
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.quantizer_propagation.graph import QuantizerPropagationStateGraph
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.insertion_point_graph import InsertionPointGraph

from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXTopKMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXNonMaxSuppressionMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import WEIGHT_LAYER_METATYPES
from nncf.experimental.onnx.hardware.fused_patterns import ONNX_HW_FUSED_PATTERNS

from collections import Counter


class NodeWithType:
    def __init__(self, name, op_type):
        self.node_name = name
        self.node_op_type = op_type


class NNCFGraphToTest:
    def __init__(self, nodes: List[NodeWithType], node_edges):
        self.nncf_graph = NNCFGraph()
        for node in nodes:
            node_name, node_op_type = node.node_name, node.node_op_type
            if 'Output' in node_name:
                metatype = OutputNoopMetatype
                node_op_type = NNCFGraphNodeType.OUTPUT_NODE
            elif 'Input' in node_name:
                metatype = InputNoopMetatype
                node_op_type = NNCFGraphNodeType.INPUT_NODE
            else:
                metatype = ONNX_OPERATION_METATYPES.get_operator_metatype_by_op_name(node_op_type)
            self.nncf_graph.add_nncf_node(node_name=node_name,
                                          node_type=node_op_type,
                                          node_metatype=metatype,
                                          layer_attributes=None)
        input_port_counter = Counter()
        output_port_counter = Counter()
        for from_node, to_nodes in node_edges.items():
            output_node_id = self.nncf_graph.get_node_by_name(from_node).node_id
            for to_node in to_nodes:
                input_node_id = self.nncf_graph.get_node_by_name(to_node).node_id
                self.nncf_graph.add_edge_between_nncf_nodes(output_node_id, input_node_id, [1],
                                                            input_port_counter[input_node_id],
                                                            output_port_counter[output_node_id], Dtype.FLOAT)
                input_port_counter[input_node_id] += 1
                output_port_counter[output_node_id] += 1


class ModelToTest1(NNCFGraphToTest):
    #              Input_1       Input_2
    #                 |             |
    #               Conv_1          |
    #                 |             |
    #              Identity_1   Identity_3
    #                  |      /     |
    #                  |     /   FC_1
    #                  |    /      |
    #                  NMS_1    Identity_4
    #                   |           |
    #                Identity_2   NMS_2
    #                   |           |
    #                 TopK_1    Identity_5
    #                   |           |
    #                Output_1    Output_2
    #
    def __init__(self):
        nodes = [NodeWithType('Input_1', 'Input'),
                 NodeWithType('Conv_1', 'Conv'),
                 NodeWithType('Identity_1', 'Identity'),
                 NodeWithType('NMS_1', 'NonMaxSuppression'),
                 NodeWithType('Identity_2', 'Identity'),
                 NodeWithType('TopK_1', 'TopK'),
                 NodeWithType('Output_1', 'Output'),
                 NodeWithType('Input_2', 'Input'),
                 NodeWithType('Identity_3', 'Identity'),
                 NodeWithType('FC_1', 'Gemm'),
                 NodeWithType('Identity_4', 'Identity'),
                 NodeWithType('NMS_2', 'NonMaxSuppression'),
                 NodeWithType('Identity_5', 'Identity'),
                 NodeWithType('Output_2', 'Output'),
                 ]
        node_edges = {'Input_1': ['Conv_1'], 'Conv_1': ['Identity_1'], 'Identity_1': ['NMS_1'], 'NMS_1': ['Identity_2'],
                      'Identity_2': ['TopK_1'], 'TopK_1': ['Output_1'], 'Input_2': ['Identity_3'],
                      'Identity_3': ['NMS_1', 'FC_1'], 'FC_1': ['Identity_4'], 'Identity_4': ['NMS_2'],
                      'NMS_2': ['Identity_5'], 'Identity_5': ['Output_2']}
        super().__init__(nodes, node_edges)
        self.reference_ignored_scopes = ['Identity_2', 'Identity_1', 'Identity_4', 'Identity_5']


class ModelToTest2(NNCFGraphToTest):
    #          Input_1
    #             |
    #           Conv_1
    #             |
    #           Identity_1
    #             |
    #            TopK_1
    #             |
    #           Identity_2
    #             |
    #           TopK_2
    #             |
    #          Identity_3
    #             |
    #           Output_1

    def __init__(self):
        nodes = [NodeWithType('Input_1', 'Input'),
                 NodeWithType('Conv_1', 'Conv'),
                 NodeWithType('Identity_1', 'Identity'),
                 NodeWithType('TopK_1', 'TopK'),
                 NodeWithType('Identity_2', 'Identity'),
                 NodeWithType('TopK_2', 'TopK'),
                 NodeWithType('Identity_3', 'Identity'),
                 NodeWithType('Output_1', 'Output')
                 ]
        node_edges = {'Input_1': ['Conv_1'], 'Conv_1': ['Identity_1'], 'Identity_1': ['TopK_1'],
                      'TopK_1': ['Identity_2'],
                      'Identity_2': ['TopK_2'], 'TopK_2': ['Identity_3'], 'Identity_3': ['Output_1']}
        super().__init__(nodes, node_edges)
        self.reference_ignored_scopes = ['Identity_2', 'Identity_1']


class ModelToTest3(NNCFGraphToTest):
    #          Input_1
    #             |
    #           Conv_1
    #             |
    #           Identity_1
    #             |
    #            TopK_1
    #             |
    #           Identity_2
    #             |      \
    #            NMS_1   Conv_2
    #             |        |
    #          Identity_3 Output_2
    #             |
    #           Output_1

    def __init__(self):
        nodes = [NodeWithType('Input_1', 'Input'),
                 NodeWithType('Conv_1', 'Conv'),
                 NodeWithType('Identity_1', 'Identity'),
                 NodeWithType('TopK_1', 'TopK'),
                 NodeWithType('Identity_2', 'Identity'),
                 NodeWithType('NMS_1', 'NonMaxSuppression'),
                 NodeWithType('Identity_3', 'Identity'),
                 NodeWithType('Output_1', 'Output'),
                 NodeWithType('Conv_2', 'Conv'),
                 NodeWithType('Output_2', 'Output')
                 ]
        node_edges = {'Input_1': ['Conv_1'], 'Conv_1': ['Identity_1'], 'Identity_1': ['TopK_1'],
                      'TopK_1': ['Identity_2'],
                      'Identity_2': ['NMS_1', 'Conv_2'], 'NMS_1': ['Identity_3'], 'Identity_3': ['Output_1'],
                      'Conv_2': ['Output_2']}
        super().__init__(nodes, node_edges)
        self.reference_ignored_scopes = ['Identity_3']


class ModelToTest4(NNCFGraphToTest):
    #          Input_1
    #             |
    #           Conv_1
    #             |
    #           Identity_1
    #             |       \
    #            TopK_1   Identity_4
    #             |      /       |
    #           Identity_2    Identity_5
    #                \       /
    #                 \     /
    #                  NMS_1
    #                    |
    #                 Identity_3
    #                    |
    #                  Output_1

    def __init__(self):
        nodes = [NodeWithType('Input_1', 'Input'),
                 NodeWithType('Conv_1', 'Conv'),
                 NodeWithType('Identity_1', 'Identity'),
                 NodeWithType('TopK_1', 'TopK'),
                 NodeWithType('Identity_2', 'Identity'),
                 NodeWithType('NMS_1', 'NonMaxSuppression'),
                 NodeWithType('Identity_3', 'Identity'),
                 NodeWithType('Output_1', 'Output'),
                 NodeWithType('Identity_4', 'Identity'),
                 NodeWithType('Identity_5', 'Identity')
                 ]
        node_edges = {'Input_1': ['Conv_1'], 'Conv_1': ['Identity_1'], 'Identity_1': ['TopK_1', 'Identity_4'],
                      'TopK_1': ['Identity_2'],
                      'Identity_2': ['NMS_1'], 'NMS_1': ['Identity_3'], 'Identity_3': ['Output_1'],
                      'Identity_4': ['Identity_2', 'Identity_5'], 'Identity_5': ['NMS_1']}
        super().__init__(nodes, node_edges)
        self.reference_ignored_scopes = ['Identity_3', 'Identity_2', 'Identity_5', 'Identity_4', 'Identity_1']


# pylint:disable=protected-access

@pytest.mark.parametrize('model_to_test', [ModelToTest1(), ModelToTest2(), ModelToTest3(), ModelToTest4()])
def test_add_ignoring_nodes_after_last_weight_node(model_to_test):
    nncf_graph = model_to_test.nncf_graph

    ip_graph = InsertionPointGraph(nncf_graph)
    pattern = ONNX_HW_FUSED_PATTERNS.get_full_pattern_graph()
    ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations(pattern)

    weight_nodes = nncf_graph.get_nodes_by_metatypes(WEIGHT_LAYER_METATYPES)
    quantizable_layer_nodes = [QuantizableWeightedLayerNode(weight_node, [QuantizerConfig()]) for weight_node in
                               weight_nodes]

    quant_prop_graph = QuantizerPropagationStateGraph(ip_graph)

    ignored_node_keys = PostprocessingNodeLocator.get_post_processing_node_keys(
        post_processing_metatypes=[ONNXTopKMetatype,
                                   ONNXNonMaxSuppressionMetatype],
        quantizable_layer_nodes=quantizable_layer_nodes,
        quant_prop_graph=quant_prop_graph)

    ignored_node_names = [nncf_graph.get_node_by_key(ignored_node_key).node_name for ignored_node_key in
                          ignored_node_keys]
    assert Counter(ignored_node_names) == Counter(model_to_test.reference_ignored_scopes)
