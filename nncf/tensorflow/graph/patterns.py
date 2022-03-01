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

from nncf.common.graph.patterns import GraphPattern
from nncf.tensorflow.graph.metatypes.tf_ops import TFAddOpMetatype
from nncf.tensorflow.graph.metatypes.tf_ops import TFRelu6OpMetatype
from nncf.tensorflow.graph.metatypes.tf_ops import TFReluOpMetatype
from nncf.tensorflow.graph.metatypes.tf_ops import TFMulOpMetatype
from nncf.tensorflow.graph.metatypes.tf_ops import TFBiasAddOpMetatype
from nncf.tensorflow.graph.metatypes.tf_ops import TFMatMulOpMetatype
from nncf.tensorflow.graph.metatypes.tf_ops import TFConv2DOpMetatype


def create_h_sigmoid_act() -> GraphPattern:
    main_pattern = GraphPattern()

    # ReLU version
    pattern = GraphPattern()

    input_pattern_node = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.NON_PATTERN_NODE_TYPE)
    add_node = pattern.add_node(label='ADD', type=TFAddOpMetatype.get_all_aliases())
    relu_node = pattern.add_node(label='RELU', type=TFReluOpMetatype.get_all_aliases())
    mul_node = pattern.add_node(label='TF_OP_MUL', type=TFMulOpMetatype.get_all_aliases())

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(add_node, relu_node)
    pattern.add_edge(relu_node, mul_node)

    main_pattern.add_pattern_alternative(pattern)

    # ReLU6 version

    pattern = GraphPattern()

    input_pattern_node = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.NON_PATTERN_NODE_TYPE)
    add_node = pattern.add_node(label='ADD', type=TFAddOpMetatype.get_all_aliases())
    relu6_node = pattern.add_node(label='RELU6', type=TFRelu6OpMetatype.get_all_aliases())
    mul_node = pattern.add_node(label='TF_OP_MUL', type=TFMulOpMetatype.get_all_aliases())

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(add_node, relu6_node)
    pattern.add_edge(relu6_node, mul_node)

    main_pattern.add_pattern_alternative(pattern)

    return main_pattern


def create_h_swish_act() -> GraphPattern:
    # TODO (vshampor): current approach with join_patterns is deficient since it does not allow to reliably
    #  connect nodes after a pattern has been joined. Along with the label and the type, the nodes created
    #  in the pattern must allow a "name" or "address" attribute, which must be a unique human readable
    #  string identifier of the node even if it has been joined multiple times, or perhaps each pattern
    #  after joining must return a list of output nodes so that these can be joined to later.
    #  Currently cannot specify h_swish in terms of h_sigmoid due to this.

    main_pattern = GraphPattern()

    # ReLU version
    pattern = GraphPattern()
    input_pattern_node = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.NON_PATTERN_NODE_TYPE)

    add_node = pattern.add_node(label='ADD', type=TFAddOpMetatype.get_all_aliases())
    relu_node = pattern.add_node(label='RELU', type=TFReluOpMetatype.get_all_aliases())
    mul_node = pattern.add_node(label='TF_OP_MUL', type=TFMulOpMetatype.get_all_aliases())

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(add_node, relu_node)
    pattern.add_edge(relu_node, mul_node)

    mul_2_node = pattern.add_node(label='MULTIPLY', type=['Multiply', 'Mul'])
    pattern.add_edge(input_pattern_node, mul_2_node)
    pattern.add_edge(mul_node, mul_2_node)
    main_pattern.add_pattern_alternative(pattern)

    # ReLU6 version
    pattern = GraphPattern()
    input_pattern_node = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.NON_PATTERN_NODE_TYPE)

    add_node = pattern.add_node(label='ADD', type=TFAddOpMetatype.get_all_aliases())
    relu6_node = pattern.add_node(label='RELU6', type=TFRelu6OpMetatype.get_all_aliases())
    mul_node = pattern.add_node(label='TF_OP_MUL', type=TFMulOpMetatype.get_all_aliases())

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(add_node, relu6_node)
    pattern.add_edge(relu6_node, mul_node)

    mul_2_node = pattern.add_node(label='MULTIPLY', type='Multiply')
    pattern.add_edge(input_pattern_node, mul_2_node)
    pattern.add_edge(mul_node, mul_2_node)
    main_pattern.add_pattern_alternative(pattern)

    return main_pattern


def create_matmul_biasadd_pattern() -> GraphPattern:
    pattern = GraphPattern()

    matmul_node = pattern.add_node(label='MATMUL', type=TFMatMulOpMetatype.get_all_aliases())
    biasadd_node = pattern.add_node(label='BIASADD', type=TFBiasAddOpMetatype.get_all_aliases())
    pattern.add_edge(matmul_node, biasadd_node)

    return pattern


def create_conv2d_biasadd_pattern() -> GraphPattern:
    pattern = GraphPattern()

    conv2d_node = pattern.add_node(label='CONV2D', type=TFConv2DOpMetatype.get_all_aliases())
    biasadd_node = pattern.add_node(label='BIASADD', type=TFBiasAddOpMetatype.get_all_aliases())
    pattern.add_edge(conv2d_node, biasadd_node)

    return pattern
