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

from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVAddMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVSubMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVMulMetatype


def create_input_preprocessing_pattern() -> GraphPattern:
    pattern = GraphPattern()

    model_input_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                             GraphPattern.METATYPE_ATTR: NNCFGraphNodeType.INPUT_NODE})
    add_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: OVAddMetatype})
    mul_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MUL',
                                     GraphPattern.METATYPE_ATTR: OVMulMetatype})

    pattern.add_edge(model_input_node_1, add_node_1)
    pattern.add_edge(add_node_1, mul_node_1)

    model_input_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                             GraphPattern.METATYPE_ATTR: NNCFGraphNodeType.INPUT_NODE})
    mul_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MUL',
                                     GraphPattern.METATYPE_ATTR: OVMulMetatype})
    add_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: OVAddMetatype})

    pattern.add_edge(model_input_node_2, mul_node_2)
    pattern.add_edge(mul_node_2, add_node_2)

    model_input_node_3 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                             GraphPattern.METATYPE_ATTR: NNCFGraphNodeType.INPUT_NODE})
    add_node_3 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: OVAddMetatype})

    pattern.add_edge(model_input_node_3, add_node_3)

    model_input_node_4 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                             GraphPattern.METATYPE_ATTR: NNCFGraphNodeType.INPUT_NODE})
    mul_node_4 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MUL',
                                     GraphPattern.METATYPE_ATTR: OVMulMetatype})

    pattern.add_edge(model_input_node_4, mul_node_4)

    model_input_node_5 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                             GraphPattern.METATYPE_ATTR: NNCFGraphNodeType.INPUT_NODE})
    mul_node_5 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SUB',
                                     GraphPattern.METATYPE_ATTR: OVMulMetatype})

    pattern.add_edge(model_input_node_5, mul_node_5)

    return pattern


def create_scale_shift() -> GraphPattern:
    pattern = GraphPattern()

    model_input_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: '*INPUT_NODE*',
                                             GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE})
    mul_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MUL',
                                     GraphPattern.METATYPE_ATTR: OVMulMetatype})
    add_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: OVAddMetatype})

    pattern.add_edge(model_input_node_1, mul_node_1)
    pattern.add_edge(mul_node_1, add_node_1)

    model_input_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: '*INPUT_NODE*',
                                             GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE})
    mul_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MUL',
                                     GraphPattern.METATYPE_ATTR: OVMulMetatype})
    add_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SUB',
                                     GraphPattern.METATYPE_ATTR: OVSubMetatype})

    pattern.add_edge(model_input_node_2, mul_node_2)
    pattern.add_edge(mul_node_2, add_node_2)

    model_input_node_3 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                             GraphPattern.METATYPE_ATTR: NNCFGraphNodeType.INPUT_NODE})
    sub_node_3 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SUB',
                                     GraphPattern.METATYPE_ATTR: OVSubMetatype})

    pattern.add_edge(model_input_node_3, sub_node_3)

    return pattern
