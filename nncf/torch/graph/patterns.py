"""
 Copyright (c) 2019-2020 Intel Corporation
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
from nncf.common.graph.patterns import create_graph_pattern_from_pattern_view
# TODO: How to use this func?
from nncf.torch.graph.version_agnostic_op_names import get_version_agnostic_name


class PatternFactory:
    def __init__(self):
        self.graph_full_pattern = None
        self.pattern_views = None

    def get_full_pattern_graph(self, pattern_views=None):
        if self.graph_full_pattern is not None and self.pattern_views == pattern_views:
            return self.graph_full_pattern
        self.pattern_views = pattern_views
        self.graph_full_pattern = get_full_pattern_graph(pattern_views)
        return self.graph_full_pattern


def get_full_pattern_graph(pattern_views=None):
    # Basic Types
    LINEAR_OPS_type = ['linear', 'conv2d', 'conv_transpose2d', 'conv3d',
                       'conv_transpose3d', 'conv1d', 'addmm']
    BN_type = ['batch_norm', 'batch_norm3d']
    POOLING_type = ['adaptive_avg_pool2d', 'adaptive_avg_pool3d', 'avg_pool2d', 'avg_pool3d']
    RELU_type = ['relu', 'relu_', 'hardtanh']
    NON_RELU_ACTIVATIONS_type = ['elu', 'elu_', 'prelu', 'sigmoid', 'gelu']
    ARITHMETIC_type = ['__iadd__', '__add__', '__mul__', '__rmul__']

    # Basic Graph Patterns
    LINEAR_OPS = GraphPattern(LINEAR_OPS_type)

    BN = GraphPattern(BN_type)

    ACTIVATIONS = GraphPattern(RELU_type + NON_RELU_ACTIVATIONS_type)

    ARITHMETIC = GraphPattern(ARITHMETIC_type)

    ANY_BN_ACT_COMBO = BN + ACTIVATIONS | ACTIVATIONS + BN | BN | ACTIVATIONS

    # Linear Types United with Swish Activation
    MUL = GraphPattern('__mul__')
    SIGMOID = GraphPattern('sigmoid')
    LINEAR_OPS_SWISH_ACTIVATION = (LINEAR_OPS + SIGMOID) * MUL | LINEAR_OPS + (BN + SIGMOID) * MUL

    FULL_PATTERN_GRAPH = LINEAR_OPS + ANY_BN_ACT_COMBO | ANY_BN_ACT_COMBO | \
                         ARITHMETIC + ANY_BN_ACT_COMBO | LINEAR_OPS_SWISH_ACTIVATION

    if pattern_views is not None:
        for pattern_view in pattern_views:
            graph_pattern = create_graph_pattern_from_pattern_view(pattern_view)
            FULL_PATTERN_GRAPH = FULL_PATTERN_GRAPH | graph_pattern

    return FULL_PATTERN_GRAPH


PATTERN_FACTORY = PatternFactory()
