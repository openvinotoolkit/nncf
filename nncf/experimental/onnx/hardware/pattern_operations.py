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

from nncf.common.graph.patterns import merge_two_types_of_operations

LINEAR_OPERATIONS = {'type': ['Conv',
                              'ConvTranspose'
                              ],
                     'label': 'LINEAR'}

BATCH_NORMALIZATION_OPERATIONS = {'type': ['BatchNormalization'],
                                  'label': 'BATCH_NORMALIZATION'}

RELU_OPERATIONS = {'type': ['Relu',
                            'Clip',
                            'LeakyRelu',
                            'ThresholdedRelu'
                            ],
                   'label': 'RELU'}

NON_RELU_ACTIVATIONS_OPERATIONS = {'type': ['Elu',
                                            'PRelu',
                                            'Sigmoid',
                                            'HardSigmoid',
                                            'HardSwish',
                                            'Tanh',
                                            'ScaledTanh'
                                            'Selu'
                                            ],
                                   'label': 'NON_RELU_ACTIVATIONS'

                                   }

ATOMIC_ACTIVATIONS_OPERATIONS = merge_two_types_of_operations(RELU_OPERATIONS,
                                                              NON_RELU_ACTIVATIONS_OPERATIONS,
                                                              'ATOMIC_ACTIVATIONS')

ARITHMETIC_OPERATIONS = {'type': ['Add',
                                  'Mul',
                                  'Div'
                                  ],
                         'label': 'ARITHMETIC'}

MATMUL_OPERATIONS = {'type': ['MatMul'
                              ],
                     'label': 'MATMUL'}
