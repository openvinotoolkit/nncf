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

pattern_dict = {
    "LINEAR_OPS" : LINEAR_OPS,
    "RELU" : RELU,
    "BN" :BN,
    "ANY_BN_RELU_COMBO":ANY_BN_RELU_COMBO,
    "POOLING" :POOLING,
    "NON_RELU_ACTIVATIONS" :NON_RELU_ACTIVATIONS,
    "SINGLE_OPS" :SINGLE_OPS,
    "ARITHMETIC":ARITHMETIC,
    "ELTWISE_UNIFORM_OPS":ELTWISE_UNIFORM_OPS,
    "MATMUL":MATMUL,
    "noop": N('noop'),
    "conv1d": N('conv1d'),
    "conv2d": N('conv2d'),
    "conv3d": N('conv3d'),
    "conv_transpose2d": N('conv_transpose2d'),
    "conv_transpose3d": N('conv_transpose3d'),
    "linear": N('linear'),
    "hardtanh": N('hardtanh'),
    "tanh": N('tanh'),
    "elu": N('elu'),
    "prelu": N('prelu'),
    "layer_norm": N('layer_norm'),
    "gelu": N('gelu'),
    "sigmoid": N('sigmoid'),
    "add": N('add'),
    "sub": N('sub'),
    "mul": N('mul'),
    "div": N('div'),
    "exp": N('exp'),
    "erf": N('erf'),
    "matmul": N('matmul'),
    "mean": N('mean'),
    "round": N('round'),
    "dropout": N('dropout'),
    "threshold": N('threshold'),
    "batch_norm": N('batch_norm'),
    "avg_pool2d": N('avg_pool2d'),
    "avg_pool3d": N('avg_pool3d'),
    "max_pool2d": N('max_pool2d'),
    "max_pool3d": N('max_pool3d'),
    "max_unpool3d": N('max_unpool3d'),
    "pad": N('pad'),
    "cat": N('cat'),
    "relu": N('relu'),
    "max": N('max'),
    "min": N('min'),
    "arange": N('arange'),
    "transpose": N('transpose'),
    "gather": N('gather'),
    "scatter": N('scatter'),
    "reshape": N('reshape'),
    "contiguous": N('contiguous'),
    "split": N('split'),
    "expand": N('expand'),
    "embedding": N('embedding'),
    "softmax": N('softmax'),
    "__lt__": N('__lt__'),
    "__le__": N('__le__'),
    "__gt__": N('__gt__'),
    "__ge__": N('__ge__'),
    "__mod__": N('__mod__'),
    "__eq__": N('__eq__'),
    "__ne__": N('__ne__'),
    "__or__": N('__or__'),
    "__xor__": N('__xor__'),
    "__and__": N('__and__'),
    "logical_not_": N('logical_not_'),
    "__pow__": N('__pow__'),
    "interpolate": N('interpolate'),
    "repeat_interleave": N('repeat_interleave'),
    "clone": N('clone')
}
