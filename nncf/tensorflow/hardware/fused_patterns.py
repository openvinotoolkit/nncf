"""
 Copyright (c) 2021 Intel Corporation
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
from nncf.common.graph.patterns import HWFusedPatterns
from nncf.tensorflow.graph.metatypes.tf_ops import TFIdentityOpMetatype
from nncf.tensorflow.graph.pattern_operations import ATOMIC_ACTIVATIONS_OPERATIONS
from nncf.tensorflow.graph.pattern_operations import BATCH_NORMALIZATION_OPERATIONS
from nncf.tensorflow.graph.pattern_operations import ELEMENTWISE_OPERATIONS
from nncf.tensorflow.graph.pattern_operations import LINEAR_OPERATIONS
from nncf.tensorflow.graph.pattern_operations import QUANTIZATION_AGNOSTIC_OPERATIONS
from nncf.tensorflow.graph.patterns import create_h_sigmoid_act
from nncf.tensorflow.graph.patterns import create_h_swish_act
from nncf.tensorflow.graph.patterns import create_matmul_biasadd_pattern
from nncf.tensorflow.graph.patterns import create_conv2d_biasadd_pattern


def _get_tf_hw_fused_patterns() -> HWFusedPatterns:
    retval = HWFusedPatterns()
    linear_ops = GraphPattern()
    linear_ops.add_node(**LINEAR_OPERATIONS)

    eltwise_ops = GraphPattern()
    eltwise_ops.add_node(**ELEMENTWISE_OPERATIONS)

    batch_norm = GraphPattern()
    batch_norm.add_node(**BATCH_NORMALIZATION_OPERATIONS)

    h_sigmoid = create_h_sigmoid_act()
    h_swish = create_h_swish_act()
    retval.register(h_sigmoid, 'H_SIGMOID', match=True)
    retval.register(h_swish, 'H_SWISH', match=True)

    matmul_biasadd = create_matmul_biasadd_pattern()
    retval.register(matmul_biasadd, 'MATMUL_BIASADD', match=True)

    conv2d_biasadd = create_conv2d_biasadd_pattern()
    retval.register(conv2d_biasadd, 'CONV2D_BIASADD', match=True)

    atomic_activations = GraphPattern()
    atomic_activations.add_node(**ATOMIC_ACTIVATIONS_OPERATIONS)
    activations = atomic_activations | h_swish | h_sigmoid
    batch_norm_activations_permutation = batch_norm + activations | activations + batch_norm
    any_bn_act_combo = batch_norm | activations | batch_norm_activations_permutation

    identity = GraphPattern()
    identity.add_node(type=TFIdentityOpMetatype.get_all_aliases(), label='IDENTITY')
    linear_ops_maybe_followed_by_identity = linear_ops | (linear_ops + identity) | matmul_biasadd | conv2d_biasadd

    agnostic_ops = GraphPattern()
    agnostic_ops.add_node(**QUANTIZATION_AGNOSTIC_OPERATIONS)
    any_ag_bn_act_combo = agnostic_ops + activations | any_bn_act_combo

    retval.register(linear_ops_maybe_followed_by_identity, name='LINEAR', match=True)
    retval.register(batch_norm_activations_permutation, name='BN_ACT_OR_ACT_BN', match=True)
    retval.register(linear_ops_maybe_followed_by_identity + any_ag_bn_act_combo, 'LINEAR + ANY_AG_BN_ACT_COMBO',
                    match=True)
    retval.register(eltwise_ops + any_ag_bn_act_combo, 'ELTWISE + ANY_AG_BN_ACT_COMBO',
                    match=True)
    return retval


TF_HW_FUSED_PATTERNS = _get_tf_hw_fused_patterns()
