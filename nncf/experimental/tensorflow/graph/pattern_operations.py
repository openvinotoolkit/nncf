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

from nncf.experimental.tensorflow.graph.metatypes.common import ELEMENTWISE_TF_OP_METATYPES
from nncf.experimental.tensorflow.graph.metatypes.common import GENERAL_CONV_TF_OP_METATYPES
from nncf.experimental.tensorflow.graph.metatypes.common import TF_OP_METATYPES_AGNOSTIC_TO_DATA_PRECISION_ONE_INPUT
from nncf.experimental.tensorflow.graph.metatypes.common import LINEAR_TF_OP_METATYPES


LINEAR_OPERATIONS = {'type': list(
    {
        *{op_type_name for m in GENERAL_CONV_TF_OP_METATYPES for op_type_name in m.get_all_aliases()},
        *{op_type_name for m in LINEAR_TF_OP_METATYPES for op_type_name in m.get_all_aliases()},
    }
),
    'label': 'LINEAR'
}


ELEMENTWISE_OPERATIONS = {'type': list(set(
    op_type_name for m in ELEMENTWISE_TF_OP_METATYPES for op_type_name in m.get_all_aliases()
)),
    'label': 'ELEMENTWISE'
}


QUANTIZATION_AGNOSTIC_OPERATIONS = {
    'type': list(
        set(
            op_type_name for m in TF_OP_METATYPES_AGNOSTIC_TO_DATA_PRECISION_ONE_INPUT \
                         for op_type_name in m.get_all_aliases()
        )
    ),
    'label': 'ELEMENTWISE'
}


BATCH_NORMALIZATION_OPERATIONS = {
    'type': [
        'FusedBatchNormV3',
    ],
    'label': 'BATCH_NORMALIZATION'
}


ATOMIC_ACTIVATIONS_OPERATIONS = {
    'type': [
        'Elu',
        'LeakyRelu',
        'Relu',
        'Relu6',
        'Selu',
        'Sigmoid',
        'Tanh'
    ],
    'label': 'ATOMIC_ACTIVATIONS'
}


POOLING_OPERATIONS = {
    'type': [
        'AvgPool',
        'AvgPool3D',
        'Mean',
    ],
    'label': 'POOLING'
}


# TODO(andrey-churkin): Update
ARITHMETIC_OPERATIONS = {'type': ['__iadd__',
                                  '__add__',
                                  '__mul__',
                                  '__rmul__'],
                         'label': 'ARITHMETIC'}
