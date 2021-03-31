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

import tensorflow as tf
import tensorflow.keras.backend as K
from beta.examples.tensorflow.common.object_detection.architecture import nn_ops


class CSPDarknet53:
    """Class to build CSPDarknet53"""

    def mish(self, x):
        return x * K.tanh(K.softplus(x))

    def DarknetConv2D_BN_Mish(self, *args, **kwargs):
        """Darknet Convolution2D followed by SyncBatchNormalization and Mish."""
        no_bias_kwargs = {'use_bias': False}
        no_bias_kwargs.update(kwargs)
        return nn_ops.compose(
            nn_ops.DarknetConv2D(*args, **no_bias_kwargs),
            tf.keras.layers.experimental.SyncBatchNormalization(),
            tf.keras.layers.Activation(self.mish))

    def csp_resblock_body(self, x, num_filters, num_blocks, all_narrow=True):
        """A series of resblocks starting with a downsampling Convolution2D"""
        # Darknet uses left and top padding instead of 'same' mode
        x = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))(x)
        x = self.DarknetConv2D_BN_Mish(num_filters, (3,3), strides=(2,2))(x)

        res_connection = self.DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(x)
        x = self.DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(x)

        for _ in range(num_blocks):
            y = nn_ops.compose(
                    self.DarknetConv2D_BN_Mish(num_filters//2, (1,1)),
                    self.DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3,3)))(x)
            x = tf.keras.layers.Add()([x,y])

        x = self.DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(x)
        x = tf.keras.layers.Concatenate()([x , res_connection])

        return self.DarknetConv2D_BN_Mish(num_filters, (1,1))(x)

    def __call__(self, x):
        """CSPDarknet53 body having 52 Convolution2D layers"""
        x = self.DarknetConv2D_BN_Mish(32, (3,3))(x)
        x = self.csp_resblock_body(x, 64, 1, False)
        x = self.csp_resblock_body(x, 128, 2)
        x = self.csp_resblock_body(x, 256, 8)
        x = self.csp_resblock_body(x, 512, 8)
        x = self.csp_resblock_body(x, 1024, 4)
        return x
