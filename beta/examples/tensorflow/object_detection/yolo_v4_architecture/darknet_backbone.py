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


import tensorflow.keras.backend as K
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Concatenate, Activation, ZeroPadding2D

from beta.examples.tensorflow.object_detection.yolo_v4_architecture.common import compose, DarknetConv2D, CustomBatchNormalization


def mish(x):
    return x * K.tanh(K.softplus(x))


def DarknetConv2D_BN_Mish(*args, **kwargs):
    """Darknet Convolution2D followed by CustomBatchNormalization and Mish."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        CustomBatchNormalization(),
        Activation(mish))


def csp_resblock_body(x, num_filters, num_blocks, all_narrow=True):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Mish(num_filters, (3,3), strides=(2,2))(x)

    res_connection = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(x)

    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Mish(num_filters//2, (1,1)),
                DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3,3)))(x)
        x = Add()([x,y])

    x = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(x)
    x = Concatenate()([x , res_connection])

    return DarknetConv2D_BN_Mish(num_filters, (1,1))(x)


def csp_darknet53_body(x):
    '''CSPDarknet53 body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Mish(32, (3,3))(x)
    x = csp_resblock_body(x, 64, 1, False)
    x = csp_resblock_body(x, 128, 2)
    x = csp_resblock_body(x, 256, 8)
    x = csp_resblock_body(x, 512, 8)
    x = csp_resblock_body(x, 1024, 4)
    return x