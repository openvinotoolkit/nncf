"""
 Copyright (c) 2020 Intel Corporation
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

import operator
from functools import reduce

from nncf.tensorflow.graph.metatypes.common import ELEMENTWISE_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import GENERAL_CONV_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT
from nncf.tensorflow.graph.metatypes.common import LINEAR_LAYER_METATYPES
from nncf.tensorflow.graph.pattern_matching import NodeExpression as N


SET_CONV_LAYERS = {layer for m in GENERAL_CONV_LAYER_METATYPES for layer in m.get_all_aliases()}
LIST_CONV_OPS = [N(layer) for layer in SET_CONV_LAYERS]
SET_LINEAR_LAYERS = {layer for m in LINEAR_LAYER_METATYPES for layer in m.get_all_aliases()}
LIST_LINEAR_OPS = [N(layer) for layer in SET_LINEAR_LAYERS]
LIST_CONV_LINEAR_OPS = LIST_CONV_OPS + LIST_LINEAR_OPS
CONV_LINEAR_OPS = reduce(operator.or_, LIST_CONV_LINEAR_OPS[1:], LIST_CONV_LINEAR_OPS[0])

SET_AGNOSTIC_LAYERS = {
    layer for m in LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT for layer in m.get_all_aliases()
}
LIST_AGNOSTIC_OPS = [N(layer) for layer in SET_AGNOSTIC_LAYERS]
AG = reduce(operator.or_, LIST_AGNOSTIC_OPS[1:], LIST_AGNOSTIC_OPS[0])

BN = N('BatchNormalization') | N('SyncBatchNormalization')

HARD_SIGMOID = (N('AddV2') + N('ReLU') + N('Mul'))
HARD_SWISH = (N('Multiply') & (HARD_SIGMOID + N('Multiply')))

KERAS_ACTIVATIONS = N('ReLU') | N('ThresholdedReLU') | N('ELU') | N('PReLU') | N('LeakyReLU') | N('Activation')
TF_ACTIVATIONS = N('Relu')
ACT = KERAS_ACTIVATIONS | TF_ACTIVATIONS | HARD_SIGMOID | HARD_SWISH

ANY_BN_ACT_COMBO = BN + ACT | ACT + BN | BN | ACT

ANY_AG_BN_ACT_COMBO = AG + ACT | ANY_BN_ACT_COMBO

POOLING = N('AveragePooling2D') | N('AveragePooling3D') | N('GlobalAveragePooling2D') | N('GlobalAveragePooling3D')

SINGLE_OPS = POOLING | N('Average') | N('LayerNormalization')

SET_ELEMENTWISE_LAYERS = {layer for m in ELEMENTWISE_LAYER_METATYPES for layer in m.get_all_aliases()}
LIST_ELEMENTWISE_OPS = [N(layer) for layer in SET_ELEMENTWISE_LAYERS]
ELEMENTWISE = reduce(operator.or_, LIST_ELEMENTWISE_OPS[1:], LIST_ELEMENTWISE_OPS[0])
