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

from collections import namedtuple
from typing import List, Optional, Type

import tensorflow as tf

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatypeRegistry
from nncf.common.hardware.opset import HWConfigOpName

WeightDef = namedtuple('WeightDef', ['weight_attr_name', 'channel_axes'])

TF_OPERATOR_METATYPES = OperatorMetatypeRegistry('operator_metatypes')


#
# NNCF Operator Metatypes Section
#
class NoopMetatype(OperatorMetatype):
    name = 'noop'


#
# TensorFlow Keras Layer Metatypes Section
#
class TFLayerMetatype(OperatorMetatype):
    keras_layer_names = []  # type: List[str]
    subtypes = []  #type: List[Type['OperatorMetatype']]

    @classmethod
    def get_subtypes(cls) -> List[Type['OperatorMetatype']]:
        return cls.subtypes

    @classmethod
    def matches(cls, layer: tf.keras.layers.Layer) -> bool:
        return layer.__class__.__name__ in cls.keras_layer_names

    @classmethod
    def determine_subtype(cls, layer: tf.keras.layers.Layer) -> Optional[Type[OperatorMetatype]]:
        matches = []
        for subtype in cls.get_subtypes():
            if subtype.matches(layer):
                subtype_matches = subtype.determine_subtype(layer)
                if subtype_matches is not None:
                    matches.extend(subtype_matches)
                else:
                    matches.append(subtype)
        if len(matches) > 1:
            raise RuntimeError('Multiple subtypes match operator call - '
                               'cannot determine single subtype.')
        if not matches:
            return None
        return matches[0]

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return cls.keras_layer_names


class TFLayerWithWeightsMetatype(TFLayerMetatype):
    weights_definition = []  # type: List[WeightDef]
    bias_attr_name = None  # type: Optional[str]


class TFDepthwiseConv1DSubLayerMetatype(TFLayerWithWeightsMetatype):
    name = 'DepthwiseConv1D(Conv1DKerasLayer)'
    keras_layer_names = ['Conv1D', 'Convolution1D']
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]

    weights_def = [WeightDef(weight_attr_name='kernel', channel_axes=-1)]
    bias_attr_name = 'bias'

    @classmethod
    def matches(cls, layer: tf.keras.layers.Layer) -> bool:
        return layer.__class__.__name__ in cls.keras_layer_names and \
               _is_depthwise_conv(layer)


@TF_OPERATOR_METATYPES.register()
class TFConv1DLayerMetatype(TFLayerWithWeightsMetatype):
    name = 'Conv1DKerasLayer'
    keras_layer_names = ['Conv1D', 'Convolution1D']
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    subtypes = [TFDepthwiseConv1DSubLayerMetatype]

    weights_def = [WeightDef(weight_attr_name='kernel', channel_axes=-1)]
    bias_attr_name = 'bias'


class TFDepthwiseConv2DSubLayerMetatype(TFLayerWithWeightsMetatype):
    name = 'DepthwiseConv2D(Conv2DKerasLayer)'
    keras_layer_names = ['Conv2D', 'Convolution2D']
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]

    weights_def = [WeightDef(weight_attr_name='kernel', channel_axes=-1)]
    bias_attr_name = 'bias'

    @classmethod
    def matches(cls, layer: tf.keras.layers.Layer) -> bool:
        return layer.__class__.__name__ in cls.keras_layer_names and \
               _is_depthwise_conv(layer)


@TF_OPERATOR_METATYPES.register()
class TFConv2DLayerMetatype(TFLayerWithWeightsMetatype):
    name = 'Conv2DKerasLayer'
    keras_layer_names = ['Conv2D', 'Convolution2D']
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    subtypes = [TFDepthwiseConv2DSubLayerMetatype]

    weights_def = [WeightDef(weight_attr_name='kernel', channel_axes=-1)]
    bias_attr_name = 'bias'


class TFDepthwiseConv3DSubLayerMetatype(TFLayerWithWeightsMetatype):
    name = 'DepthwiseConv3D(Conv3DKerasLayer)'
    keras_layer_names = ['Conv3D', 'Convolution3D']
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]

    weights_def = [WeightDef(weight_attr_name='kernel', channel_axes=-1)]
    bias_attr_name = 'bias'

    @classmethod
    def matches(cls, layer: tf.keras.layers.Layer) -> bool:
        return layer.__class__.__name__ in cls.keras_layer_names and \
               _is_depthwise_conv(layer)


@TF_OPERATOR_METATYPES.register()
class TFConv3DLayerMetatype(TFLayerWithWeightsMetatype):
    name = 'Conv3DKerasLayer'
    keras_layer_names = ['Conv3D', 'Convolution3D']
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    subtypes = [TFDepthwiseConv3DSubLayerMetatype]

    weights_def = [WeightDef(weight_attr_name='kernel', channel_axes=-1)]
    bias_attr_name = 'bias'


@TF_OPERATOR_METATYPES.register()
class TFDepthwiseConv2DLayerMetatype(TFLayerWithWeightsMetatype):
    name = 'DepthwiseConv2DKerasLayer'
    keras_layer_names = ['DepthwiseConv2D']
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]

    weights_def = [WeightDef(weight_attr_name='depthwise_kernel', channel_axes=(2, 3))]
    bias_attr_name = 'bias'


@TF_OPERATOR_METATYPES.register()
class TFConv1DTransposeLayerMetatype(TFLayerWithWeightsMetatype):
    name = 'Conv1DTransposeKerasLayer'
    keras_layer_names = ['Conv1DTranspose', 'Convolution1DTranspose']
    hw_config_names = [HWConfigOpName.CONVOLUTION]

    weights_def = [WeightDef(weight_attr_name='kernel', channel_axes=-2)]
    bias_attr_name = 'bias'


@TF_OPERATOR_METATYPES.register()
class TFConv2DTransposeLayerMetatype(TFLayerWithWeightsMetatype):
    name = 'Conv2DTransposeKerasLayer'
    keras_layer_names = ['Conv2DTranspose', 'Convolution2DTranspose']
    hw_config_names = [HWConfigOpName.CONVOLUTION]

    weights_def = [WeightDef(weight_attr_name='kernel', channel_axes=-2)]
    bias_attr_name = 'bias'


@TF_OPERATOR_METATYPES.register()
class TFConv3DTransposeLayerMetatype(TFLayerWithWeightsMetatype):
    name = 'Conv3DTransposeKerasLayer'
    keras_layer_names = ['Conv3DTranspose', 'Convolution3DTranspose']
    hw_config_names = [HWConfigOpName.CONVOLUTION]

    weights_def = [WeightDef(weight_attr_name='kernel', channel_axes=-2)]
    bias_attr_name = 'bias'


@TF_OPERATOR_METATYPES.register()
class TFDenseLayerMetatype(TFLayerWithWeightsMetatype):
    name = 'DenseKerasLayer'
    keras_layer_names = ['Dense']
    hw_config_names = [HWConfigOpName.MATMUL]

    weights_def = [WeightDef(weight_attr_name='kernel', channel_axes=-1)]
    bias_attr_name = 'bias'


@TF_OPERATOR_METATYPES.register()
class TFBatchNormalizationLayerMetatype(TFLayerWithWeightsMetatype):
    name = 'BatchNormalizationKerasLayer'
    keras_layer_names = ['BatchNormalization', 'SyncBatchNormalization']

    weights_def = [WeightDef(weight_attr_name='gamma', channel_axes=0)]
    bias_attr_name = 'beta'


@TF_OPERATOR_METATYPES.register()
class TFSeparableConv1DLayerMetatype(TFLayerWithWeightsMetatype):
    name = 'SeparableConv1DKerasLayer'
    keras_layer_names = ['SeparableConv1D', 'SeparableConvolution1D']

    weights_def = [
        WeightDef(weight_attr_name='depthwise_kernel', channel_axes=(1, 2)),
        WeightDef(weight_attr_name='pointwise_kernel', channel_axes=-1),
    ]
    bias_attr_name = 'bias'


@TF_OPERATOR_METATYPES.register()
class TFSeparableConv2DLayerMetatype(TFLayerWithWeightsMetatype):
    name = 'SeparableConv2DKerasLayer'
    keras_layer_names = ['SeparableConv2D', 'SeparableConvolution2D']

    weights_def = [
        WeightDef(weight_attr_name='depthwise_kernel', channel_axes=(2, 3)),
        WeightDef(weight_attr_name='pointwise_kernel', channel_axes=-1),
    ]
    bias_attr_name = 'bias'


@TF_OPERATOR_METATYPES.register()
class TFEmbeddingLayerMetatype(TFLayerWithWeightsMetatype):
    name = 'EmbeddingKerasLayer'
    keras_layer_names = ['Embedding']

    weights_def = [WeightDef(weight_attr_name='embeddings', channel_axes=None)]


@TF_OPERATOR_METATYPES.register()
class TFLocallyConnected1DLayerMetatype(TFLayerWithWeightsMetatype):
    name = 'LocallyConnected1DKerasLayer'
    keras_layer_names = ['LocallyConnected1D']

    weights_def = [WeightDef(weight_attr_name='kernel', channel_axes=None)]


@TF_OPERATOR_METATYPES.register()
class TFLocallyConnected2DLayerMetatype(TFLayerWithWeightsMetatype):
    name = 'LocallyConnected2DKerasLayer'
    keras_layer_names = ['LocallyConnected2D']

    weights_def = [WeightDef(weight_attr_name='kernel', channel_axes=None)]


@TF_OPERATOR_METATYPES.register()
class TFCropping1DLayerMetatype(TFLayerMetatype):
    name = 'Cropping1DKerasLayer'
    keras_layer_names = ['Cropping1D']
    hw_config_names = [HWConfigOpName.CROP]


@TF_OPERATOR_METATYPES.register()
class TFCropping2DLayerMetatype(TFLayerMetatype):
    name = 'Cropping2DKerasLayer'
    keras_layer_names = ['Cropping2D']
    hw_config_names = [HWConfigOpName.CROP]


@TF_OPERATOR_METATYPES.register()
class TFCropping3DLayerMetatype(TFLayerMetatype):
    name = 'Cropping3DKerasLayer'
    keras_layer_names = ['Cropping3D']
    hw_config_names = [HWConfigOpName.CROP]


@TF_OPERATOR_METATYPES.register()
class TFFlattenLayerMetatype(TFLayerMetatype):
    name = 'FlattenKerasLayer'
    keras_layer_names = ['Flatten']
    hw_config_names = [HWConfigOpName.FLATTEN]


@TF_OPERATOR_METATYPES.register()
class TFGlobalMaxPooling1DLayerMetatype(TFLayerMetatype):
    name = 'GlobalMaxPooling1DKerasLayer'
    keras_layer_names = ['GlobalMaxPool1D', 'GlobalMaxPooling1D']
    hw_config_names = [HWConfigOpName.MAXPOOL]


@TF_OPERATOR_METATYPES.register()
class TFGlobalMaxPooling2DLayerMetatype(TFLayerMetatype):
    name = 'GlobalMaxPooling2DKerasLayer'
    keras_layer_names = ['GlobalMaxPool2D', 'GlobalMaxPooling2D']
    hw_config_names = [HWConfigOpName.MAXPOOL]


@TF_OPERATOR_METATYPES.register()
class TFGlobalMaxPooling3DLayerMetatype(TFLayerMetatype):
    name = 'GlobalMaxPooling3DKerasLayer'
    keras_layer_names = ['GlobalMaxPool3D', 'GlobalMaxPooling3D']
    hw_config_names = [HWConfigOpName.MAXPOOL]


@TF_OPERATOR_METATYPES.register()
class TFMaxPooling1DLayerMetatype(TFLayerMetatype):
    name = 'MaxPooling1DKerasLayer'
    keras_layer_names = ['MaxPool1D', 'MaxPooling1D']
    hw_config_names = [HWConfigOpName.MAXPOOL]


@TF_OPERATOR_METATYPES.register()
class TFMaxPooling2DLayerMetatype(TFLayerMetatype):
    name = 'MaxPooling2DKerasLayer'
    keras_layer_names = ['MaxPool2D', 'MaxPooling2D']
    hw_config_names = [HWConfigOpName.MAXPOOL]


@TF_OPERATOR_METATYPES.register()
class TFMaxPooling3DLayerMetatype(TFLayerMetatype):
    name = 'MaxPooling3DKerasLayer'
    keras_layer_names = ['MaxPool3D', 'MaxPooling3D']
    hw_config_names = [HWConfigOpName.MAXPOOL]


@TF_OPERATOR_METATYPES.register()
class TFRepeatVectorLayerMetatype(TFLayerMetatype):
    name = 'RepeatVectorKerasLayer'
    keras_layer_names = ['RepeatVector']
    hw_config_names = [HWConfigOpName.TILE]


@TF_OPERATOR_METATYPES.register()
class TFReshapeLayerMetatype(TFLayerMetatype):
    name = 'ReshapeKerasLayer'
    keras_layer_names = ['Reshape']
    hw_config_names = [HWConfigOpName.RESHAPE]


@TF_OPERATOR_METATYPES.register()
class TFZeroPadding1DLayerMetatype(TFLayerMetatype):
    name = 'ZeroPadding1DKerasLayer'
    keras_layer_names = ['ZeroPadding1D']
    hw_config_names = [HWConfigOpName.PAD]


@TF_OPERATOR_METATYPES.register()
class TFZeroPadding2DLayerMetatype(TFLayerMetatype):
    name = 'ZeroPadding2DKerasLayer'
    keras_layer_names = ['ZeroPadding2D']
    hw_config_names = [HWConfigOpName.PAD]


@TF_OPERATOR_METATYPES.register()
class TFZeroPadding3DLayerMetatype(TFLayerMetatype):
    name = 'ZeroPadding3DKerasLayer'
    keras_layer_names = ['ZeroPadding3D']
    hw_config_names = [HWConfigOpName.PAD]


@TF_OPERATOR_METATYPES.register()
class TFUpSampling1DLayerMetatype(TFLayerMetatype):
    # Split->Concat pattern
    name = 'UpSampling1DKerasLayer'
    keras_layer_names = ['UpSampling1D']


@TF_OPERATOR_METATYPES.register()
class TFUpSampling2DLayerMetatype(TFLayerMetatype):
    name = 'UpSampling2DKerasLayer'
    keras_layer_names = ['UpSampling2D']
    hw_config_names = [HWConfigOpName.INTERPOLATE]


@TF_OPERATOR_METATYPES.register()
class TFUpSampling3DLayerMetatype(TFLayerMetatype):
    name = 'UpSampling3DKerasLayer'
    keras_layer_names = ['UpSampling3D']
    hw_config_names = [HWConfigOpName.INTERPOLATE]


@TF_OPERATOR_METATYPES.register()
class TFConcatenateLayerMetatype(TFLayerMetatype):
    name = 'ConcatenateKerasLayer'
    keras_layer_names = ['Concatenate']
    hw_config_names = [HWConfigOpName.CONCAT]


@TF_OPERATOR_METATYPES.register()
class TFAddLayerMetatype(TFLayerMetatype):
    name = 'AddKerasLayer'
    keras_layer_names = ['Add']
    hw_config_names = [HWConfigOpName.ADD]


@TF_OPERATOR_METATYPES.register()
class TFSubtractLayerMetatype(TFLayerMetatype):
    name = 'SubtractKerasLayer'
    keras_layer_names = ['Subtract']
    hw_config_names = [HWConfigOpName.SUBTRACT]


@TF_OPERATOR_METATYPES.register()
class TFMultiplyLayerMetatype(TFLayerMetatype):
    name = 'MultiplyKerasLayer'
    keras_layer_names = ['Multiply']
    hw_config_names = [HWConfigOpName.MULTIPLY]


@TF_OPERATOR_METATYPES.register()
class TFAveragePooling1DLayerMetatype(TFLayerMetatype):
    name = 'AveragePooling1DKerasLayer'
    keras_layer_names = ['AveragePooling1D', 'AvgPool1D']
    hw_config_names = [HWConfigOpName.AVGPOOL]


@TF_OPERATOR_METATYPES.register()
class TFAveragePooling2DLayerMetatype(TFLayerMetatype):
    name = 'AveragePooling2DKerasLayer'
    keras_layer_names = ['AveragePooling2D', 'AvgPool2D']
    hw_config_names = [HWConfigOpName.AVGPOOL]


@TF_OPERATOR_METATYPES.register()
class TFAveragePooling3DLayerMetatype(TFLayerMetatype):
    name = 'AveragePooling3DKerasLayer'
    keras_layer_names = ['AveragePooling3D', 'AvgPool3D']
    hw_config_names = [HWConfigOpName.AVGPOOL]


@TF_OPERATOR_METATYPES.register()
class TFGlobalAveragePooling1DLayerMetatype(TFLayerMetatype):
    name = 'GlobalAveragePooling1DKerasLayer'
    keras_layer_names = ['GlobalAveragePooling1D', 'GlobalAvgPool1D']
    hw_config_names = [HWConfigOpName.AVGPOOL]


@TF_OPERATOR_METATYPES.register()
class TFGlobalAveragePooling2DLayerMetatype(TFLayerMetatype):
    name = 'GlobalAveragePooling2DKerasLayer'
    keras_layer_names = ['GlobalAveragePooling2D', 'GlobalAvgPool2D']
    hw_config_names = [HWConfigOpName.AVGPOOL]


@TF_OPERATOR_METATYPES.register()
class TFGlobalAveragePooling3DLayerMetatype(TFLayerMetatype):
    name = 'GlobalAveragePooling3DKerasLayer'
    keras_layer_names = ['GlobalAveragePooling3D', 'GlobalAvgPool3D']
    hw_config_names = [HWConfigOpName.AVGPOOL]


@TF_OPERATOR_METATYPES.register()
class TFReLULayerMetatype(TFLayerMetatype):
    name = 'ReLUKerasLayer'
    keras_layer_names = ['ReLU']


@TF_OPERATOR_METATYPES.register()
class TFThresholdedReLULayerMetatype(TFLayerMetatype):
    name = 'ThresholdedReLUKerasLayer'
    keras_layer_names = ['ThresholdedReLU']


@TF_OPERATOR_METATYPES.register()
class TFELULayerMetatype(TFLayerMetatype):
    name = 'ELUKerasLayer'
    keras_layer_names = ['ELU']


@TF_OPERATOR_METATYPES.register()
class TFPReLULayerMetatype(TFLayerMetatype):
    name = 'PReLUKerasLayer'
    keras_layer_names = ['PReLU']


@TF_OPERATOR_METATYPES.register()
class TFLeakyReLULayerMetatype(TFLayerMetatype):
    name = 'LeakyReLUKerasLayer'
    keras_layer_names = ['LeakyReLU']


@TF_OPERATOR_METATYPES.register()
class TFSoftmaxLayerMetatype(TFLayerMetatype):
    name = 'SoftmaxKerasLayer'
    keras_layer_names = ['Softmax']


@TF_OPERATOR_METATYPES.register()
class TFActivationLayerMetatype(TFLayerMetatype):
    name = 'ActivationKerasLayer'
    keras_layer_names = ['Activation']


@TF_OPERATOR_METATYPES.register()
class TFAverageLayerMetatype(TFLayerMetatype):
    name = 'AverageKerasLayer'
    keras_layer_names = ['Average']


@TF_OPERATOR_METATYPES.register()
class TFLayerNormalizationLayerMetatype(TFLayerMetatype):
    name = 'LayerNormalizationKerasLayer'
    keras_layer_names = ['LayerNormalization']


@TF_OPERATOR_METATYPES.register()
class TFInputLayerMetatype(TFLayerMetatype):
    name = 'InputLayer'
    keras_layer_names = ['InputLayer']


@TF_OPERATOR_METATYPES.register()
class TFDropoutLayerMetatype(TFLayerMetatype):
    name = 'DropoutKerasLayer'
    keras_layer_names = ['Dropout']


@TF_OPERATOR_METATYPES.register()
class TFLambdaLayerMetatype(TFLayerMetatype):
    name = 'LambdaKerasLayer'
    keras_layer_names = ['Lambda']


#
# TensorFlow  Raw Operations Section
#
TF_RAW_OPERATOR_METATYPES = OperatorMetatypeRegistry('raw_operator_metatypes')

class TFOpMetatype(OperatorMetatype):
    op_names = []  # type: List[str]

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return cls.op_names


@TF_RAW_OPERATOR_METATYPES.register()
class TFIdentityOpMetatype(TFOpMetatype):
    name = 'IdentityOp'
    op_names = ['Identity']


@TF_RAW_OPERATOR_METATYPES.register()
class TFPackOpMetatype(TFOpMetatype):
    # Unsqueezes->Concat pattern
    name = 'PackOp'
    op_names = ['Pack']


@TF_RAW_OPERATOR_METATYPES.register()
class TFPadOpMetatype(TFOpMetatype):
    name = 'PadOp'
    op_names = ['Pad']
    hw_config_names = [HWConfigOpName.PAD]


@TF_RAW_OPERATOR_METATYPES.register()
class TFStridedSliceOpMetatype(TFOpMetatype):
    name = 'StridedSliceOp'
    op_names = ['StridedSlice']
    hw_config_names = [HWConfigOpName.STRIDEDSLICE]


@TF_RAW_OPERATOR_METATYPES.register()
class TFConcatOpMetatype(TFOpMetatype):
    name = 'ConcatOp'
    op_names = ['Concat', 'ConcatV2']
    hw_config_names = [HWConfigOpName.CONCAT]


@TF_RAW_OPERATOR_METATYPES.register()
class TFAddOpMetatype(TFOpMetatype):
    name = 'AddOp'
    op_names = ['Add', 'AddV2']
    hw_config_names = [HWConfigOpName.ADD]


@TF_RAW_OPERATOR_METATYPES.register()
class TFSubOpMetatype(TFOpMetatype):
    name = 'SubOp'
    op_names = ['Sub']
    hw_config_names = [HWConfigOpName.SUBTRACT]


@TF_RAW_OPERATOR_METATYPES.register()
class TFMulOpMetatype(TFOpMetatype):
    name = 'MulOp'
    op_names = ['Mul']
    hw_config_names = [HWConfigOpName.MULTIPLY]


@TF_RAW_OPERATOR_METATYPES.register()
class TFAvgPoolOpMetatype(TFOpMetatype):
    name = 'AvgPoolOp'
    op_names = ['AvgPool']
    hw_config_names = [HWConfigOpName.AVGPOOL]


@TF_RAW_OPERATOR_METATYPES.register()
class TFAvgPool3DOpMetatype(TFOpMetatype):
    name = 'AvgPool3DOp'
    op_names = ['AvgPool3D']
    hw_config_names = [HWConfigOpName.AVGPOOL]


@TF_RAW_OPERATOR_METATYPES.register()
class TFReluOpMetatype(TFOpMetatype):
    name = 'ReluOp'
    op_names = ['Relu']


@TF_RAW_OPERATOR_METATYPES.register()
class TFRelu6OpMetatype(TFOpMetatype):
    name = 'Relu6Op'
    op_names = ['Relu6']


@TF_RAW_OPERATOR_METATYPES.register()
class TFMatMulOpMetatype(TFOpMetatype):
    name = 'MatMulOp'
    op_names = ['MatMul']


@TF_OPERATOR_METATYPES.register()
class TFTensorFlowOpLayerMetatype(TFLayerMetatype):
    name = 'TensorFlowOpKerasLayer'
    keras_layer_names = ['TensorFlowOpLayer']

    @classmethod
    def get_subtypes(cls) -> List[Type['OperatorMetatype']]:
        return list(TF_RAW_OPERATOR_METATYPES.registry_dict.values())

    @classmethod
    def determine_subtype(cls, layer: tf.keras.layers.Layer) -> Optional[Type[OperatorMetatype]]:
        return TF_RAW_OPERATOR_METATYPES.get_operator_metatype_by_op_name(layer.node_def.op)


def _is_depthwise_conv(layer: tf.keras.layers.Layer) -> bool:
    channel_axis = -1 - layer.rank \
        if layer.data_format == 'channels_first' else -1

    if layer.input_shape.dims[channel_axis].value is None:
        raise ValueError('The channel dimension of the inputs '
                         'should be defined. Found `None`.')

    input_channels = int(layer.input_shape[channel_axis])

    return input_channels == layer.groups and input_channels > 1
