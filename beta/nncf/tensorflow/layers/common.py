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

WEIGHT_ATTR_NAME = 'weight_attr_name'
BIAS_ATTR_NAME = 'bias_attr_name'
CHANNEL_AXES = 'channel_axes'

KERAS_LAYERS_WITH_WEIGHTS = {
    'Conv1D': {WEIGHT_ATTR_NAME: 'kernel', CHANNEL_AXES: -1, BIAS_ATTR_NAME: 'bias'},
    'Conv2D': {WEIGHT_ATTR_NAME: 'kernel', CHANNEL_AXES: -1, BIAS_ATTR_NAME: 'bias'},
    'Conv3D': {WEIGHT_ATTR_NAME: 'kernel', CHANNEL_AXES: -1, BIAS_ATTR_NAME: 'bias'},
    'DepthwiseConv2D': {WEIGHT_ATTR_NAME: 'depthwise_kernel', CHANNEL_AXES: (2, 3), BIAS_ATTR_NAME: 'bias'},
    'Conv1DTranspose': {WEIGHT_ATTR_NAME: 'kernel', CHANNEL_AXES: -2, BIAS_ATTR_NAME: 'bias'},
    'Conv2DTranspose': {WEIGHT_ATTR_NAME: 'kernel', CHANNEL_AXES: -2, BIAS_ATTR_NAME: 'bias'},
    'Conv3DTranspose': {WEIGHT_ATTR_NAME: 'kernel', CHANNEL_AXES: -2, BIAS_ATTR_NAME: 'bias'},
    'Dense': {WEIGHT_ATTR_NAME: 'kernel', CHANNEL_AXES: -1, BIAS_ATTR_NAME: 'bias'}
}

TF_LAYERS_WITH_WEIGHTS = {}

LAYERS_WITH_WEIGHTS = {}
LAYERS_WITH_WEIGHTS.update(KERAS_LAYERS_WITH_WEIGHTS)
LAYERS_WITH_WEIGHTS.update(TF_LAYERS_WITH_WEIGHTS)

SPECIAL_LAYERS_WITH_WEIGHTS = {
    'BatchNormalization': {WEIGHT_ATTR_NAME: 'gamma', BIAS_ATTR_NAME: 'beta'}
}

ALL_LAYERS_WITH_WEIGHTS = {}
ALL_LAYERS_WITH_WEIGHTS.update(LAYERS_WITH_WEIGHTS)
ALL_LAYERS_WITH_WEIGHTS.update(SPECIAL_LAYERS_WITH_WEIGHTS)

DECONV_LAYERS = [
    'Conv1DTranspose',
    'Conv2DTranspose',
    'Conv3DTranspose'
]

GENERAL_CONV_LAYERS = [
    'Conv1D',
    'Conv2D',
    'Conv3D',
    'DepthwiseConv2D',
    'Conv1DTranspose',
    'Conv2DTranspose',
    'Conv3DTranspose'
]

LINEAR_LAYERS = [
    'Dense'
]

KERAS_LAYERS_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT = [
    'Cropping1D',
    'Cropping2D',
    'Cropping3D',
    'Flatten',
    'GlobalMaxPool1D',
    'GlobalMaxPool2D',
    'GlobalMaxPool3D',
    'GlobalMaxPooling1D',
    'GlobalMaxPooling2D',
    'GlobalMaxPooling3D',
    'MaxPool1D',
    'MaxPool2D',
    'MaxPool3D',
    'MaxPooling1D',
    'MaxPooling2D',
    'MaxPooling3D',
    'RepeatVector',
    'Reshape',
    'UpSampling2D'
    'ZeroPadding1D',
    'ZeroPadding2D',
    'ZeroPadding3D'
]

TF_LAYERS_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT = [
    'Identity',
    'Pack',
    'Pad',
    'StridedSlice',
]

LAYERS_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT = \
    KERAS_LAYERS_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT + TF_LAYERS_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT

KERAS_LAYERS_AGNOSTIC_TO_DATA_PRECISION_WITH_MULTIPLE_INPUTS = [
    'Concatenate'
]

TF_LAYERS_AGNOSTIC_TO_DATA_PRECISION_WITH_MULTIPLE_INPUTS = [
    'ConcatV2'
]

LAYERS_AGNOSTIC_TO_DATA_PRECISION_WITH_MULTIPLE_INPUTS = \
    KERAS_LAYERS_AGNOSTIC_TO_DATA_PRECISION_WITH_MULTIPLE_INPUTS \
    + TF_LAYERS_AGNOSTIC_TO_DATA_PRECISION_WITH_MULTIPLE_INPUTS

LAYERS_AGNOSTIC_TO_DATA_PRECISION = \
    LAYERS_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT + LAYERS_AGNOSTIC_TO_DATA_PRECISION_WITH_MULTIPLE_INPUTS

KERAS_ELEMENTWISE_LAYERS = [
    'Add',
    'Multiply',
    'Rescaling'
]

TF_ELEMENTWISE_LAYERS = [
    'AddV2',
    'Mul'
]

ELEMENTWISE_LAYERS = KERAS_ELEMENTWISE_LAYERS + TF_ELEMENTWISE_LAYERS
