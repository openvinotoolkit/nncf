#
# Copyright 2020-2021 Intel Corporation.
#
# LEGAL NOTICE: Your use of this software and any required dependent software
# (the "Software Package") is subject to the terms and conditions of
# the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
# which may also include notices, disclaimers, or license terms for
# third party or open source software included in or with the Software Package,
# and your use indicates your acceptance of all such terms. Please refer
# to the "third-party-programs.txt" or other similarly-named text file
# included with the Software Package for additional details.

def merge_two_types_of_operations(first_op, second_op, label):
    res = {'type': first_op['type']}
    res['type'].extend(second_op['type'])
    res['label'] = label
    return res


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
                                            'gelu'],
                                   'label': 'NON_RELU_ACTIVATIONS'}

ACTIVATIONS_OPERATIONS = merge_two_types_of_operations(RELU_OPERATIONS,
                                                       NON_RELU_ACTIVATIONS_OPERATIONS,
                                                       'ACTIVATIONS')

ARITHMETIC_OPERATIONS = {'type': ['__iadd__',
                                  '__add__',
                                  '__mul__',
                                  '__rmul__'],
                         'label': 'ARITHMETIC'}

# This type may be useful in the future
# pylint: disable=unused-variable
POOLING_OPERATIONS = {'type': ['adaptive_avg_pool2d',
                               'adaptive_avg_pool3d',
                               'avg_pool2d',
                               'avg_pool3d'],
                      'label': 'POOLING'}
