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
from nncf.common.graph.patterns import merge_two_types_of_operations

LINEAR_OPERATIONS = {'type': ['linear',
                              'conv1d',
                              'conv2d',
                              'conv3d',
                              'conv_transpose1d',
                              'conv_transpose2d',
                              'conv_transpose3d',
                              'addmm'
                              ],
                     'label': 'LINEAR'}

BATCH_NORMALIZATION_OPERATIONS = {'type': ['batch_norm',
                                           'batch_norm1d',
                                           'batch_norm2d',
                                           'batch_norm3d'
                                           ],
                                  'label': 'BATCH_NORMALIZATION'
                                  }

GROUP_NORMALIZATION_OPERATIONS = {'type': ['group_norm'],
                                  'label': 'GROUP_NORMALIZATION'}

RELU_OPERATIONS = {'type': ['relu',
                            'relu_',
                            'hardtanh'
                            ],
                   'label': 'RELU'
                   }

NON_RELU_ACTIVATIONS_OPERATIONS = {'type': ['elu',
                                            'elu_',
                                            'prelu',
                                            'sigmoid',
                                            'gelu',
                                            'silu',
                                            'hardsigmoid',
                                            'hardswish'],
                                   'label': 'NON_RELU_ACTIVATIONS'}

ATOMIC_ACTIVATIONS_OPERATIONS = merge_two_types_of_operations(RELU_OPERATIONS,
                                                              NON_RELU_ACTIVATIONS_OPERATIONS,
                                                              'ATOMIC_ACTIVATIONS')

ARITHMETIC_OPERATIONS = {'type': ['__iadd__',
                                  '__add__',
                                  '__mul__',
                                  '__rmul__',
                                  '__truediv__'],
                         'label': 'ARITHMETIC'}

# This type may be useful in the future
# pylint: disable=unused-variable
POOLING_OPERATIONS = {'type': ['adaptive_avg_pool2d',
                               'adaptive_avg_pool3d',
                               'avg_pool2d',
                               'avg_pool3d'],
                      'label': 'POOLING'}

MATMUL_OPERATIONS = {'type': ['bmm',
                              'matmul'
                              ],
                     'label': 'MATMUL'}
