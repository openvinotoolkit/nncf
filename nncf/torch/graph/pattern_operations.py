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

LINEAR_OPERATIONS = ['linear',
                     'conv1d',
                     'conv2d',
                     'conv3d',
                     'conv_transpose1d',
                     'conv_transpose2d',
                     'conv_transpose3d',
                     'addmm'
                     ]

BATCH_NORMALIZATION_OPERATIONS = ['batch_norm',
                                  'batch_norm1d',
                                  'batch_norm2d',
                                  'batch_norm3d']
RELU_OPERATIONS = ['RELU',
                   'hardtanh']

NON_RELU_ACTIVATIONS_OPERATIONS = ['elu',
                                   'elu_',
                                   'prelu',
                                   'sigmoid',
                                   'gelu']

ACTIVATIONS_OPERATIONS = RELU_OPERATIONS + NON_RELU_ACTIVATIONS_OPERATIONS

ARITHMETIC_OPERATIONS = ['__iadd__',
                         '__add__',
                         '__mul__',
                         '__rmul__']

# This type may be useful in the future
# pylint: disable=unused-variable
POOLING_OPERATIONS = ['adaptive_avg_pool2d',
                      'adaptive_avg_pool3d',
                      'avg_pool2d',
                      'avg_pool3d']
