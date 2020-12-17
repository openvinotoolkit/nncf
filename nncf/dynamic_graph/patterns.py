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

from nncf.dynamic_graph.graph import NNCFNodeExpression as N
from nncf.dynamic_graph.version_agnostic_op_names import VersionAgnosticNames

LINEAR_OPS = N('linear') | N('conv2d') | N('conv_transpose2d') | N('conv3d') | \
             N('conv_transpose3d') | N('conv1d') | N('addmm')

RELU = N(VersionAgnosticNames.RELU) | N('hardtanh')

BN = N('batch_norm') | N('batch_norm3d')

POOLING = N('adaptive_avg_pool2d') | N('adaptive_avg_pool3d') | N('avg_pool2d') | N('avg_pool3d')

NON_RELU_ACTIVATIONS = N('elu') | N('elu_') | N('prelu') | N('sigmoid') | N('gelu')

ACTIVATIONS = RELU | NON_RELU_ACTIVATIONS

ANY_BN_ACT_COMBO = BN + ACTIVATIONS | ACTIVATIONS + BN | BN | ACTIVATIONS

SINGLE_OPS = ACTIVATIONS | POOLING | N('mean') | N('layer_norm')

ARITHMETIC = N('__iadd__') | N('__add__') | N('__mul__') | N('__rmul__')

ELTWISE_UNIFORM_OPS = BN | RELU | ACTIVATIONS

MATMUL = N('matmul') | N('bmm')

FUSED_CONV_BN = N('Conv2dBN2d')
