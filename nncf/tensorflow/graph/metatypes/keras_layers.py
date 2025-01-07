# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import namedtuple
from typing import List, Optional, Type

import tensorflow as tf

import nncf
from nncf.common.graph.operator_metatypes import INPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatypeRegistry
from nncf.common.hardware.opset import HWConfigOpName
from nncf.tensorflow.graph.metatypes.tf_ops import TF_OPERATION_METATYPES

WeightDef = namedtuple("WeightDef", ["weight_attr_name", "channel_axes"])

KERAS_LAYER_METATYPES = OperatorMetatypeRegistry("keras_layer_metatypes")


class TFLayerMetatype(OperatorMetatype):
    keras_layer_names: List[str] = []
    subtypes: List[Type[OperatorMetatype]] = []

    @classmethod
    def get_subtypes(cls) -> List[Type[OperatorMetatype]]:
        return cls.subtypes

    @classmethod
    def determine_subtype(cls, layer: tf.keras.layers.Layer) -> Optional[Type[OperatorMetatype]]:
        return cls._determine_subtype(layer)

    @classmethod
    def determine_subtype_wrapped_layer(
        cls, layer: tf.keras.layers.Layer, wrapper: Optional[tf.keras.layers.Wrapper] = None
    ) -> Optional[Type[OperatorMetatype]]:
        return cls._determine_subtype(layer, wrapper)

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return cls.keras_layer_names

    @classmethod
    def matches(cls, layer: tf.keras.layers.Layer, _: Optional[tf.keras.layers.Wrapper] = None) -> bool:
        return layer.__class__.__name__ in cls.keras_layer_names

    @classmethod
    def _determine_subtype(
        cls, layer: tf.keras.layers.Layer, wrapper: Optional[tf.keras.layers.Wrapper] = None
    ) -> Optional[Type[OperatorMetatype]]:
        matches = []
        for subtype in cls.get_subtypes():
            if subtype.matches(layer, wrapper):
                subtype_matches = subtype.determine_subtype_wrapped_layer(layer, wrapper)
                if subtype_matches is not None:
                    matches.extend(subtype_matches)
                else:
                    matches.append(subtype)
        if len(matches) > 1:
            raise nncf.InternalError("Multiple subtypes match operator call - cannot determine single subtype.")
        if not matches:
            return None
        return matches[0]


class TFLayerWithWeightsMetatype(TFLayerMetatype):
    weight_definitions: List[WeightDef] = []
    bias_attr_name: Optional[str] = None


@KERAS_LAYER_METATYPES.register()
@NOOP_METATYPES.register()
class TFLayerNoopMetatype(TFLayerMetatype):
    name = "noop"

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return [cls.name]


@KERAS_LAYER_METATYPES.register(is_subtype=True)
class TFDepthwiseConv1DSubLayerMetatype(TFLayerWithWeightsMetatype):
    name = "DepthwiseConv1D(Conv1DKerasLayer)"
    keras_layer_names = ["Conv1D", "Convolution1D"]
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]

    weight_definitions = [WeightDef(weight_attr_name="kernel", channel_axes=-1)]
    bias_attr_name = "bias"

    @classmethod
    def matches(cls, layer: tf.keras.layers.Layer, wrapper: Optional[tf.keras.layers.Wrapper] = None) -> bool:
        return layer.__class__.__name__ in cls.keras_layer_names and _is_depthwise_conv(layer, wrapper)


@KERAS_LAYER_METATYPES.register()
class TFConv1DLayerMetatype(TFLayerWithWeightsMetatype):
    name = "Conv1DKerasLayer"
    keras_layer_names = ["Conv1D", "Convolution1D"]
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    subtypes = [TFDepthwiseConv1DSubLayerMetatype]

    weight_definitions = [WeightDef(weight_attr_name="kernel", channel_axes=-1)]
    bias_attr_name = "bias"


@KERAS_LAYER_METATYPES.register(is_subtype=True)
class TFDepthwiseConv2DSubLayerMetatype(TFLayerWithWeightsMetatype):
    name = "DepthwiseConv2D(Conv2DKerasLayer)"
    keras_layer_names = ["Conv2D", "Convolution2D"]
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]

    weight_definitions = [WeightDef(weight_attr_name="kernel", channel_axes=-1)]
    bias_attr_name = "bias"

    @classmethod
    def matches(cls, layer: tf.keras.layers.Layer, wrapper: Optional[tf.keras.layers.Wrapper] = None) -> bool:
        return layer.__class__.__name__ in cls.keras_layer_names and _is_depthwise_conv(layer, wrapper)


@KERAS_LAYER_METATYPES.register()
class TFConv2DLayerMetatype(TFLayerWithWeightsMetatype):
    name = "Conv2DKerasLayer"
    keras_layer_names = ["Conv2D", "Convolution2D"]
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    subtypes = [TFDepthwiseConv2DSubLayerMetatype]

    weight_definitions = [WeightDef(weight_attr_name="kernel", channel_axes=-1)]
    bias_attr_name = "bias"


@KERAS_LAYER_METATYPES.register(is_subtype=True)
class TFDepthwiseConv3DSubLayerMetatype(TFLayerWithWeightsMetatype):
    name = "DepthwiseConv3D(Conv3DKerasLayer)"
    keras_layer_names = ["Conv3D", "Convolution3D"]
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]

    weight_definitions = [WeightDef(weight_attr_name="kernel", channel_axes=-1)]
    bias_attr_name = "bias"

    @classmethod
    def matches(cls, layer: tf.keras.layers.Layer, wrapper: Optional[tf.keras.layers.Wrapper] = None) -> bool:
        return layer.__class__.__name__ in cls.keras_layer_names and _is_depthwise_conv(layer, wrapper)


@KERAS_LAYER_METATYPES.register()
class TFConv3DLayerMetatype(TFLayerWithWeightsMetatype):
    name = "Conv3DKerasLayer"
    keras_layer_names = ["Conv3D", "Convolution3D"]
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    subtypes = [TFDepthwiseConv3DSubLayerMetatype]

    weight_definitions = [WeightDef(weight_attr_name="kernel", channel_axes=-1)]
    bias_attr_name = "bias"


@KERAS_LAYER_METATYPES.register()
class TFDepthwiseConv2DLayerMetatype(TFLayerWithWeightsMetatype):
    name = "DepthwiseConv2DKerasLayer"
    keras_layer_names = ["DepthwiseConv2D"]
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]

    weight_definitions = [WeightDef(weight_attr_name="depthwise_kernel", channel_axes=(2, 3))]
    bias_attr_name = "bias"


@KERAS_LAYER_METATYPES.register()
class TFConv1DTransposeLayerMetatype(TFLayerWithWeightsMetatype):
    name = "Conv1DTransposeKerasLayer"
    keras_layer_names = ["Conv1DTranspose", "Convolution1DTranspose"]
    hw_config_names = [HWConfigOpName.CONVOLUTION]

    weight_definitions = [WeightDef(weight_attr_name="kernel", channel_axes=-2)]
    bias_attr_name = "bias"


@KERAS_LAYER_METATYPES.register()
class TFConv2DTransposeLayerMetatype(TFLayerWithWeightsMetatype):
    name = "Conv2DTransposeKerasLayer"
    keras_layer_names = ["Conv2DTranspose", "Convolution2DTranspose"]
    hw_config_names = [HWConfigOpName.CONVOLUTION]

    weight_definitions = [WeightDef(weight_attr_name="kernel", channel_axes=-2)]
    bias_attr_name = "bias"


@KERAS_LAYER_METATYPES.register()
class TFConv3DTransposeLayerMetatype(TFLayerWithWeightsMetatype):
    name = "Conv3DTransposeKerasLayer"
    keras_layer_names = ["Conv3DTranspose", "Convolution3DTranspose"]
    hw_config_names = [HWConfigOpName.CONVOLUTION]

    weight_definitions = [WeightDef(weight_attr_name="kernel", channel_axes=-2)]
    bias_attr_name = "bias"


@KERAS_LAYER_METATYPES.register()
class TFDenseLayerMetatype(TFLayerWithWeightsMetatype):
    name = "DenseKerasLayer"
    keras_layer_names = ["Dense"]
    hw_config_names = [HWConfigOpName.MATMUL]

    weight_definitions = [WeightDef(weight_attr_name="kernel", channel_axes=-1)]
    bias_attr_name = "bias"


@KERAS_LAYER_METATYPES.register()
class TFBatchNormalizationLayerMetatype(TFLayerWithWeightsMetatype):
    name = "BatchNormalizationKerasLayer"
    keras_layer_names = ["BatchNormalization", "SyncBatchNormalization"]

    weight_definitions = [WeightDef(weight_attr_name="gamma", channel_axes=0)]
    bias_attr_name = "beta"


@KERAS_LAYER_METATYPES.register()
class TFSeparableConv1DLayerMetatype(TFLayerWithWeightsMetatype):
    name = "SeparableConv1DKerasLayer"
    keras_layer_names = ["SeparableConv1D", "SeparableConvolution1D"]

    weight_definitions = [
        WeightDef(weight_attr_name="depthwise_kernel", channel_axes=(1, 2)),
        WeightDef(weight_attr_name="pointwise_kernel", channel_axes=-1),
    ]
    bias_attr_name = "bias"


@KERAS_LAYER_METATYPES.register()
class TFSeparableConv2DLayerMetatype(TFLayerWithWeightsMetatype):
    name = "SeparableConv2DKerasLayer"
    keras_layer_names = ["SeparableConv2D", "SeparableConvolution2D"]

    weight_definitions = [
        WeightDef(weight_attr_name="depthwise_kernel", channel_axes=(2, 3)),
        WeightDef(weight_attr_name="pointwise_kernel", channel_axes=-1),
    ]
    bias_attr_name = "bias"


@KERAS_LAYER_METATYPES.register()
class TFEmbeddingLayerMetatype(TFLayerWithWeightsMetatype):
    name = "EmbeddingKerasLayer"
    keras_layer_names = ["Embedding"]

    weight_definitions = [WeightDef(weight_attr_name="embeddings", channel_axes=None)]


@KERAS_LAYER_METATYPES.register()
class TFLocallyConnected1DLayerMetatype(TFLayerWithWeightsMetatype):
    name = "LocallyConnected1DKerasLayer"
    keras_layer_names = ["LocallyConnected1D"]

    weight_definitions = [WeightDef(weight_attr_name="kernel", channel_axes=None)]


@KERAS_LAYER_METATYPES.register()
class TFLocallyConnected2DLayerMetatype(TFLayerWithWeightsMetatype):
    name = "LocallyConnected2DKerasLayer"
    keras_layer_names = ["LocallyConnected2D"]

    weight_definitions = [WeightDef(weight_attr_name="kernel", channel_axes=None)]


@KERAS_LAYER_METATYPES.register()
class TFCropping1DLayerMetatype(TFLayerMetatype):
    name = "Cropping1DKerasLayer"
    keras_layer_names = ["Cropping1D"]
    hw_config_names = [HWConfigOpName.CROP]


@KERAS_LAYER_METATYPES.register()
class TFCropping2DLayerMetatype(TFLayerMetatype):
    name = "Cropping2DKerasLayer"
    keras_layer_names = ["Cropping2D"]
    hw_config_names = [HWConfigOpName.CROP]


@KERAS_LAYER_METATYPES.register()
class TFCropping3DLayerMetatype(TFLayerMetatype):
    name = "Cropping3DKerasLayer"
    keras_layer_names = ["Cropping3D"]
    hw_config_names = [HWConfigOpName.CROP]


@KERAS_LAYER_METATYPES.register()
class TFFlattenLayerMetatype(TFLayerMetatype):
    name = "FlattenKerasLayer"
    keras_layer_names = ["Flatten"]
    hw_config_names = [HWConfigOpName.FLATTEN]


@KERAS_LAYER_METATYPES.register()
class TFGlobalMaxPooling1DLayerMetatype(TFLayerMetatype):
    name = "GlobalMaxPooling1DKerasLayer"
    keras_layer_names = ["GlobalMaxPool1D", "GlobalMaxPooling1D"]
    hw_config_names = [HWConfigOpName.MAXPOOL]


@KERAS_LAYER_METATYPES.register()
class TFGlobalMaxPooling2DLayerMetatype(TFLayerMetatype):
    name = "GlobalMaxPooling2DKerasLayer"
    keras_layer_names = ["GlobalMaxPool2D", "GlobalMaxPooling2D"]
    hw_config_names = [HWConfigOpName.MAXPOOL]


@KERAS_LAYER_METATYPES.register()
class TFGlobalMaxPooling3DLayerMetatype(TFLayerMetatype):
    name = "GlobalMaxPooling3DKerasLayer"
    keras_layer_names = ["GlobalMaxPool3D", "GlobalMaxPooling3D"]
    hw_config_names = [HWConfigOpName.MAXPOOL]


@KERAS_LAYER_METATYPES.register()
class TFMaxPooling1DLayerMetatype(TFLayerMetatype):
    name = "MaxPooling1DKerasLayer"
    keras_layer_names = ["MaxPool1D", "MaxPooling1D"]
    hw_config_names = [HWConfigOpName.MAXPOOL]


@KERAS_LAYER_METATYPES.register()
class TFMaxPooling2DLayerMetatype(TFLayerMetatype):
    name = "MaxPooling2DKerasLayer"
    keras_layer_names = ["MaxPool2D", "MaxPooling2D"]
    hw_config_names = [HWConfigOpName.MAXPOOL]


@KERAS_LAYER_METATYPES.register()
class TFMaxPooling3DLayerMetatype(TFLayerMetatype):
    name = "MaxPooling3DKerasLayer"
    keras_layer_names = ["MaxPool3D", "MaxPooling3D"]
    hw_config_names = [HWConfigOpName.MAXPOOL]


@KERAS_LAYER_METATYPES.register()
class TFRepeatVectorLayerMetatype(TFLayerMetatype):
    name = "RepeatVectorKerasLayer"
    keras_layer_names = ["RepeatVector"]
    hw_config_names = [HWConfigOpName.TILE]


@KERAS_LAYER_METATYPES.register()
class TFReshapeLayerMetatype(TFLayerMetatype):
    name = "ReshapeKerasLayer"
    keras_layer_names = ["Reshape"]
    hw_config_names = [HWConfigOpName.RESHAPE]


@KERAS_LAYER_METATYPES.register()
class TFPermuteLayerMetatype(TFLayerMetatype):
    name = "PermuteKerasLayer"
    keras_layer_names = ["Permute"]
    hw_config_names = [HWConfigOpName.TRANSPOSE]


@KERAS_LAYER_METATYPES.register()
class TFZeroPadding1DLayerMetatype(TFLayerMetatype):
    name = "ZeroPadding1DKerasLayer"
    keras_layer_names = ["ZeroPadding1D"]
    hw_config_names = [HWConfigOpName.PAD]


@KERAS_LAYER_METATYPES.register()
class TFZeroPadding2DLayerMetatype(TFLayerMetatype):
    name = "ZeroPadding2DKerasLayer"
    keras_layer_names = ["ZeroPadding2D"]
    hw_config_names = [HWConfigOpName.PAD]


@KERAS_LAYER_METATYPES.register()
class TFZeroPadding3DLayerMetatype(TFLayerMetatype):
    name = "ZeroPadding3DKerasLayer"
    keras_layer_names = ["ZeroPadding3D"]
    hw_config_names = [HWConfigOpName.PAD]


@KERAS_LAYER_METATYPES.register()
class TFUpSampling1DLayerMetatype(TFLayerMetatype):
    # Split->Concat pattern
    name = "UpSampling1DKerasLayer"
    keras_layer_names = ["UpSampling1D"]


@KERAS_LAYER_METATYPES.register()
class TFUpSampling2DLayerMetatype(TFLayerMetatype):
    name = "UpSampling2DKerasLayer"
    keras_layer_names = ["UpSampling2D"]
    hw_config_names = [HWConfigOpName.INTERPOLATE]


@KERAS_LAYER_METATYPES.register()
class TFUpSampling3DLayerMetatype(TFLayerMetatype):
    name = "UpSampling3DKerasLayer"
    keras_layer_names = ["UpSampling3D"]
    hw_config_names = [HWConfigOpName.INTERPOLATE]


@KERAS_LAYER_METATYPES.register()
class TFConcatenateLayerMetatype(TFLayerMetatype):
    name = "ConcatenateKerasLayer"
    keras_layer_names = ["Concatenate"]
    hw_config_names = [HWConfigOpName.CONCAT]


@KERAS_LAYER_METATYPES.register()
class TFAddLayerMetatype(TFLayerMetatype):
    name = "AddKerasLayer"
    keras_layer_names = ["Add"]
    hw_config_names = [HWConfigOpName.ADD]


@KERAS_LAYER_METATYPES.register()
class TFSubtractLayerMetatype(TFLayerMetatype):
    name = "SubtractKerasLayer"
    keras_layer_names = ["Subtract"]
    hw_config_names = [HWConfigOpName.SUBTRACT]


@KERAS_LAYER_METATYPES.register()
class TFMultiplyLayerMetatype(TFLayerMetatype):
    name = "MultiplyKerasLayer"
    keras_layer_names = ["Multiply"]
    hw_config_names = [HWConfigOpName.MULTIPLY]


@KERAS_LAYER_METATYPES.register()
class TFRescalingLayerMetatype(TFLayerMetatype):
    name = "RescalingKerasLayer"
    keras_layer_names = ["Rescaling"]


@KERAS_LAYER_METATYPES.register()
class TFAveragePooling1DLayerMetatype(TFLayerMetatype):
    name = "AveragePooling1DKerasLayer"
    keras_layer_names = ["AveragePooling1D", "AvgPool1D"]
    hw_config_names = [HWConfigOpName.AVGPOOL]


@KERAS_LAYER_METATYPES.register()
class TFAveragePooling2DLayerMetatype(TFLayerMetatype):
    name = "AveragePooling2DKerasLayer"
    keras_layer_names = ["AveragePooling2D", "AvgPool2D"]
    hw_config_names = [HWConfigOpName.AVGPOOL]


@KERAS_LAYER_METATYPES.register()
class TFAveragePooling3DLayerMetatype(TFLayerMetatype):
    name = "AveragePooling3DKerasLayer"
    keras_layer_names = ["AveragePooling3D", "AvgPool3D"]
    hw_config_names = [HWConfigOpName.AVGPOOL]


@KERAS_LAYER_METATYPES.register()
class TFGlobalAveragePooling1DLayerMetatype(TFLayerMetatype):
    name = "GlobalAveragePooling1DKerasLayer"
    keras_layer_names = ["GlobalAveragePooling1D", "GlobalAvgPool1D"]
    hw_config_names = [HWConfigOpName.AVGPOOL]


@KERAS_LAYER_METATYPES.register()
class TFGlobalAveragePooling2DLayerMetatype(TFLayerMetatype):
    name = "GlobalAveragePooling2DKerasLayer"
    keras_layer_names = ["GlobalAveragePooling2D", "GlobalAvgPool2D"]
    hw_config_names = [HWConfigOpName.AVGPOOL]


@KERAS_LAYER_METATYPES.register()
class TFGlobalAveragePooling3DLayerMetatype(TFLayerMetatype):
    name = "GlobalAveragePooling3DKerasLayer"
    keras_layer_names = ["GlobalAveragePooling3D", "GlobalAvgPool3D"]
    hw_config_names = [HWConfigOpName.AVGPOOL]


@KERAS_LAYER_METATYPES.register()
class TFReLULayerMetatype(TFLayerMetatype):
    name = "ReLUKerasLayer"
    keras_layer_names = ["ReLU"]


@KERAS_LAYER_METATYPES.register()
class TFThresholdedReLULayerMetatype(TFLayerMetatype):
    name = "ThresholdedReLUKerasLayer"
    keras_layer_names = ["ThresholdedReLU"]


@KERAS_LAYER_METATYPES.register()
class TFELULayerMetatype(TFLayerMetatype):
    name = "ELUKerasLayer"
    keras_layer_names = ["ELU"]


@KERAS_LAYER_METATYPES.register()
class TFPReLULayerMetatype(TFLayerMetatype):
    name = "PReLUKerasLayer"
    keras_layer_names = ["PReLU"]


@KERAS_LAYER_METATYPES.register()
class TFLeakyReLULayerMetatype(TFLayerMetatype):
    name = "LeakyReLUKerasLayer"
    keras_layer_names = ["LeakyReLU"]


@KERAS_LAYER_METATYPES.register()
class TFSoftmaxLayerMetatype(TFLayerMetatype):
    name = "SoftmaxKerasLayer"
    keras_layer_names = ["Softmax"]


@KERAS_LAYER_METATYPES.register()
class TFActivationLayerMetatype(TFLayerMetatype):
    name = "ActivationKerasLayer"
    keras_layer_names = ["Activation"]


@KERAS_LAYER_METATYPES.register()
class TFAverageLayerMetatype(TFLayerMetatype):
    name = "AverageKerasLayer"
    keras_layer_names = ["Average"]


@KERAS_LAYER_METATYPES.register()
class TFLayerNormalizationLayerMetatype(TFLayerMetatype):
    name = "LayerNormalizationKerasLayer"
    keras_layer_names = ["LayerNormalization"]


@KERAS_LAYER_METATYPES.register()
@INPUT_NOOP_METATYPES.register()
class TFInputLayerMetatype(TFLayerMetatype):
    name = "InputLayer"
    keras_layer_names = ["InputLayer"]


@KERAS_LAYER_METATYPES.register()
class TFDropoutLayerMetatype(TFLayerMetatype):
    name = "DropoutKerasLayer"
    keras_layer_names = ["Dropout"]


@KERAS_LAYER_METATYPES.register()
class TFLambdaLayerMetatype(TFLayerMetatype):
    name = "LambdaKerasLayer"
    keras_layer_names = ["Lambda"]


@KERAS_LAYER_METATYPES.register()
class TFTensorFlowOpLayerMetatype(TFLayerMetatype):
    name = "TensorFlowOpKerasLayer"
    keras_layer_names = ["TensorFlowOpLayer"]

    @classmethod
    def get_subtypes(cls) -> List[Type[OperatorMetatype]]:
        return list(TF_OPERATION_METATYPES.registry_dict.values())

    @classmethod
    def determine_subtype(
        cls, layer: tf.keras.layers.Layer, wrapper: Optional[tf.keras.layers.Wrapper] = None
    ) -> Optional[Type[OperatorMetatype]]:
        return TF_OPERATION_METATYPES.get_operator_metatype_by_op_name(layer.node_def.op)


@KERAS_LAYER_METATYPES.register()
class TFOpLambdaMetatype(TFLayerMetatype):
    name = "TFOpLambdaKerasLayer"
    keras_layer_names = ["TFOpLambda"]

    @classmethod
    def get_subtypes(cls) -> List[Type[OperatorMetatype]]:
        return list(TF_OPERATION_METATYPES.registry_dict.values())

    @classmethod
    def determine_subtype(
        cls, layer: tf.keras.layers.Layer, wrapper: Optional[tf.keras.layers.Wrapper] = None
    ) -> Optional[Type[OperatorMetatype]]:
        return TF_OPERATION_METATYPES.get_operator_metatype_by_op_name(layer.symbol)


@KERAS_LAYER_METATYPES.register()
class TFSlicingOpLambdaMetatype(TFLayerMetatype):
    name = "SlicingOpLambdaKerasLayer"
    keras_layer_names = ["SlicingOpLambda"]

    @classmethod
    def get_subtypes(cls) -> List[Type[OperatorMetatype]]:
        return list(TF_OPERATION_METATYPES.registry_dict.values())

    @classmethod
    def determine_subtype(
        cls, layer: tf.keras.layers.Layer, wrapper: Optional[tf.keras.layers.Wrapper] = None
    ) -> Optional[Type[OperatorMetatype]]:
        return TF_OPERATION_METATYPES.get_operator_metatype_by_op_name(layer.symbol)


@KERAS_LAYER_METATYPES.register()
class TFNNCFWrapperLayerMetatype(TFLayerMetatype):
    name = "NNCFWrapperLayer"
    keras_layer_names = ["NNCFWrapper"]

    @classmethod
    def get_subtypes(cls) -> List[Type[OperatorMetatype]]:
        return list(KERAS_LAYER_METATYPES.registry_dict.values())

    @classmethod
    def determine_subtype(cls, layer: tf.keras.layers.Layer) -> Optional[Type[OperatorMetatype]]:
        unwrapped_layer = layer.layer
        unwrapped_layer_metatype = KERAS_LAYER_METATYPES.get_operator_metatype_by_op_name(
            unwrapped_layer.__class__.__name__
        )
        subtype = unwrapped_layer_metatype.determine_subtype_wrapped_layer(unwrapped_layer, layer)
        if subtype is not None:
            return subtype
        return unwrapped_layer_metatype

    @classmethod
    def _determine_subtype(
        cls, layer: tf.keras.layers.Layer, wrapper: Optional[tf.keras.layers.Wrapper] = None
    ) -> Optional[Type[OperatorMetatype]]:
        unwrapped_layer = layer.layer
        return super()._determine_subtype(unwrapped_layer, wrapper)


def _is_depthwise_conv(layer: tf.keras.layers.Layer, wrapper: Optional[tf.keras.layers.Wrapper] = None) -> bool:
    channel_axis = -1 - layer.rank if layer.data_format == "channels_first" else -1

    channels = (
        layer.get_input_shape_at(0)[channel_axis] if wrapper is None else wrapper.get_input_shape_at(0)[channel_axis]
    )

    if channels is None:
        raise ValueError("The channel dimension of the inputs should be defined. Found `None`.")

    input_channels = int(channels)

    return input_channels == layer.groups and input_channels > 1
