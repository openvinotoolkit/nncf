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

from tensorflow.keras.layers import Concatenate, MaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU

from beta.examples.tensorflow.object_detection.yolo_v4_architecture.common import compose, DarknetConv2D, CustomBatchNormalization


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by CustomBatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        CustomBatchNormalization(),
        LeakyReLU(alpha=0.1))


def Spp_Conv2D_BN_Leaky(x, num_filters):
    y1 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(x)
    y2 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(x)
    y3 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(x)

    y = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))([y3, y2, y1, x])
    return y


def make_yolo_head(x, num_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)

    return x


def make_yolo_spp_head(x, num_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)

    x = Spp_Conv2D_BN_Leaky(x, num_filters)

    x = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)

    return x


def yolo4_predictions(feature_maps, feature_channel_nums, num_anchors, num_classes):
    f1, f2, f3 = feature_maps
    f1_channel_num, f2_channel_num, f3_channel_num = feature_channel_nums

    #feature map 1 head (19x19 for 608 input)
    x1 = make_yolo_spp_head(f1, f1_channel_num//2)

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)

    x2 = DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1))(f2)
    x2 = Concatenate()([x2, x1_upsample])

    #feature map 2 head (38x38 for 608 input)
    x2 = make_yolo_head(x2, f2_channel_num//2)

    #upsample fpn merge for feature map 2 & 3
    x2_upsample = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1)),
            UpSampling2D(2))(x2)

    x3 = DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1))(f3)
    x3 = Concatenate()([x3, x2_upsample])

    #feature map 3 head & output (76x76 for 608 input)
    #x3, y3 = make_last_layers(x3, f3_channel_num//2, num_anchors*(num_classes+5))
    x3 = make_yolo_head(x3, f3_channel_num//2)
    y3 = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_3'))(x3)

    #downsample fpn merge for feature map 3 & 2
    x3_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (3,3), strides=(2,2)))(x3)

    x2 = Concatenate()([x3_downsample, x2])

    #feature map 2 output (38x38 for 608 input)
    #x2, y2 = make_last_layers(x2, 256, num_anchors*(num_classes+5))
    x2 = make_yolo_head(x2, f2_channel_num//2)
    y2 = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_2'))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            DarknetConv2D_BN_Leaky(f1_channel_num//2, (3,3), strides=(2,2)))(x2)

    x1 = Concatenate()([x2_downsample, x1])

    #feature map 1 output (19x19 for 608 input)
    #x1, y1 = make_last_layers(x1, f1_channel_num//2, num_anchors*(num_classes+5))
    x1 = make_yolo_head(x1, f1_channel_num//2)
    y1 = compose(
            DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_1'))(x1)

    return y1, y2, y3