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

from typing import List

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatypeRegistry
from nncf.common.hardware.opset import HWConfigOpName

TF_OPERATION_METATYPES = OperatorMetatypeRegistry('tf_operation_metatypes')


class TFOpMetatype(OperatorMetatype):
    op_names = []  # type: List[str]

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return cls.op_names


class OpWeightDef:
    """
    Contains information about the weight of operation.
    """

    def __init__(self, port_id: int, channel_axes):
        """
        Initializes a definition of the weight.

        :param port_id: Zero-based argument number of the operation to
            which this weight tensor corresponds.
        :param channel_axes: Channel axes for weight tensor.
        """
        # TODO(andrey-churkin): Seems like we can determine the port id
        # dynamically during the NNCF Graph building.
        self.port_id = port_id
        self.channel_axes = channel_axes


class TFOpWithWeightsMetatype(TFOpMetatype):
    weight_definitions = []  # type: List[OpWeightDef]


@TF_OPERATION_METATYPES.register()
class TFNoopMetatype(OperatorMetatype):
    name = 'noop'
    op_names = ['noop']


@TF_OPERATION_METATYPES.register()
class TFIdentityOpMetatype(TFOpMetatype):
    name = 'IdentityOp'
    op_names = ['Identity', 'identity']


@TF_OPERATION_METATYPES.register()
class TFPackOpMetatype(TFOpMetatype):
    # Unsqueezes->Concat pattern
    name = 'PackOp'
    op_names = ['Pack', 'stack']


@TF_OPERATION_METATYPES.register()
class TFUnPackOpMetatype(TFOpMetatype):
    name = 'UnPackOp'
    op_names = ['Unpack', 'unstack']


@TF_OPERATION_METATYPES.register()
class TFPadOpMetatype(TFOpMetatype):
    name = 'PadOp'
    op_names = ['Pad', 'compat.v1.pad', 'pad']
    hw_config_names = [HWConfigOpName.PAD]


@TF_OPERATION_METATYPES.register()
class TFStridedSliceOpMetatype(TFOpMetatype):
    name = 'StridedSliceOp'
    op_names = ['StridedSlice', '__operators__.getitem']
    hw_config_names = [HWConfigOpName.STRIDEDSLICE]


@TF_OPERATION_METATYPES.register()
class TFConcatOpMetatype(TFOpMetatype):
    name = 'ConcatOp'
    op_names = ['Concat', 'ConcatV2', 'concat']
    hw_config_names = [HWConfigOpName.CONCAT]


@TF_OPERATION_METATYPES.register()
class TFAddOpMetatype(TFOpMetatype):
    name = 'AddOp'
    op_names = ['Add', 'AddV2', '__operators__.add']
    hw_config_names = [HWConfigOpName.ADD]


@TF_OPERATION_METATYPES.register()
class TFSubOpMetatype(TFOpMetatype):
    name = 'SubOp'
    op_names = ['Sub', 'math.subtract']
    hw_config_names = [HWConfigOpName.SUBTRACT]


@TF_OPERATION_METATYPES.register()
class TFMulOpMetatype(TFOpMetatype):
    name = 'MulOp'
    op_names = ['Mul', 'math.multiply']
    hw_config_names = [HWConfigOpName.MULTIPLY]


@TF_OPERATION_METATYPES.register()
class TFAvgPoolOpMetatype(TFOpMetatype):
    name = 'AvgPoolOp'
    op_names = ['AvgPool', 'nn.avg_pool']
    hw_config_names = [HWConfigOpName.AVGPOOL]


@TF_OPERATION_METATYPES.register()
class TFAvgPool3DOpMetatype(TFOpMetatype):
    name = 'AvgPool3DOp'
    op_names = ['AvgPool3D']
    hw_config_names = [HWConfigOpName.AVGPOOL]


@TF_OPERATION_METATYPES.register()
class TFReluOpMetatype(TFOpMetatype):
    name = 'ReluOp'
    op_names = ['Relu', 'ReLU', 'nn.relu']


@TF_OPERATION_METATYPES.register()
class TFRelu6OpMetatype(TFOpMetatype):
    name = 'Relu6Op'
    op_names = ['Relu6']


@TF_OPERATION_METATYPES.register()
class TFMatMulOpMetatype(TFOpWithWeightsMetatype):
    name = 'MatMulOp'
    op_names = ['MatMul', 'linalg.matmul']
    weight_definitions = [OpWeightDef(port_id=1, channel_axes=-1)]
    hw_config_names = [HWConfigOpName.MATMUL]


@TF_OPERATION_METATYPES.register()
class TFConv2DOpMetatype(TFOpWithWeightsMetatype):
    name = 'Conv2DOp'
    op_names = ['Conv2D']
    weight_definitions = [OpWeightDef(port_id=1, channel_axes=-1)]
    hw_config_names = [HWConfigOpName.CONVOLUTION]


@TF_OPERATION_METATYPES.register()
class TFConv3DOpMetatype(TFOpWithWeightsMetatype):
    name = 'Conv3DOp'
    op_names = ['Conv3D']
    weight_definitions = [OpWeightDef(port_id=1, channel_axes=-1)]
    hw_config_names = [HWConfigOpName.CONVOLUTION]


@TF_OPERATION_METATYPES.register()
class TFDepthwiseConv2dNativeOpMetatype(TFOpWithWeightsMetatype):
    name = 'DepthwiseConv2dNativeOp'
    op_names = ['DepthwiseConv2dNative']
    weight_definitions = [OpWeightDef(port_id=1, channel_axes=[2, 3])]
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]


@TF_OPERATION_METATYPES.register()
class TFQuantizedConv2DOpMetatype(TFOpMetatype):
    name = 'QuantizedConv2DOp'
    op_names = ['QuantizedConv2D']


@TF_OPERATION_METATYPES.register()
class TFReshapeOpMetatype(TFOpMetatype):
    name = 'ReshapeOp'
    op_names = ['Reshape', 'reshape']
    hw_config_names = [HWConfigOpName.RESHAPE]


@TF_OPERATION_METATYPES.register()
class TFExpandDimsOpMetatype(TFOpMetatype):
    name = 'ExpandDimsOp'
    op_names = ['ExpandDims', 'expand_dims']


@TF_OPERATION_METATYPES.register()
class TFSplitOpMetatype(TFOpMetatype):
    name = 'SplitOp'
    op_names = ['Split', 'split']
    hw_config_names = [HWConfigOpName.SPLIT]


@TF_OPERATION_METATYPES.register()
class TFMinimumOpMetatype(TFOpMetatype):
    name = 'MinimumOp'
    op_names = ['Minimum', 'math.minimum']
    hw_config_names = [HWConfigOpName.MINIMUM]


@TF_OPERATION_METATYPES.register()
class TFMaximumOpMetatype(TFOpMetatype):
    name = 'MaximumOp'
    op_names = ['Maximum', 'math.maximum']
    hw_config_names = [HWConfigOpName.MAXIMUM]


@TF_OPERATION_METATYPES.register()
class TFExpOpMetatype(TFOpMetatype):
    name = 'ExpOp'
    op_names = ['Exp', 'math.exp']


@TF_OPERATION_METATYPES.register()
class TFPlaceholderOpMetatype(TFOpMetatype):
    name = 'PlaceholderOp'
    op_names = ['Placeholder']


@TF_OPERATION_METATYPES.register()
class TFShapeOpMetatype(TFOpMetatype):
    name = 'ShapeOp'
    op_names = ['Shape', 'compat.v1.shape']


@TF_OPERATION_METATYPES.register()
class TFBiasAddOpMetatype(TFOpMetatype):
    name = 'BiasAddOp'
    op_names = ['BiasAdd']
    hw_config_names = [HWConfigOpName.ADD]


@TF_OPERATION_METATYPES.register()
class TFMeanOpMetatype(TFOpMetatype):
    name = 'MeanOp'
    op_names = ['Mean', 'math.reduce_mean']
    hw_config_names = [
        HWConfigOpName.REDUCEMEAN,
        HWConfigOpName.AVGPOOL,
    ]


@TF_OPERATION_METATYPES.register()
class TFFusedBatchNormV3OpMetatype(TFOpMetatype):
    name = 'FusedBatchNormV3Op'
    op_names = ['FusedBatchNormV3']


@TF_OPERATION_METATYPES.register()
class TFSqueezeOpMetatype(TFOpMetatype):
    name = 'SqueezeOp'
    op_names = ['Squeeze', 'squeeze']
    hw_config_names = [HWConfigOpName.SQUEEZE]


@TF_OPERATION_METATYPES.register()
class TFSigmoidOpMetatype(TFOpMetatype):
    name = 'SigmoidOp'
    op_names = ['Sigmoid', 'math.sigmoid']


@TF_OPERATION_METATYPES.register()
class TFCombinedNonMaxSuppressionOpMetatype(TFOpMetatype):
    name = 'CombinedNonMaxSuppressionOp'
    op_names = ['CombinedNonMaxSuppression', 'image.combined_non_max_suppression']


@TF_OPERATION_METATYPES.register()
class TFTopKV2OpMetatype(TFOpMetatype):
    name = 'TopKV2Op'
    op_names = ['TopKV2', 'math.top_k']


@TF_OPERATION_METATYPES.register()
class TFGatherOpMetatype(TFOpMetatype):
    name = 'GatherOp'
    op_names = ['GatherNd', 'GatherV2', 'compat.v1.gather', 'compat.v1.gather_nd']


@TF_OPERATION_METATYPES.register()
class TFPowOpMetatype(TFOpMetatype):
    name = 'PowOp'
    op_names = ['Pow', 'math.pow', 'Sqrt', 'math.sqrt']
    hw_config_names = [HWConfigOpName.POWER]


@TF_OPERATION_METATYPES.register()
class TFTrueDivOpMetatype(TFOpMetatype):
    name = 'TrueDivOp'
    op_names = ['RealDiv', 'math.divide', 'math.truediv']
    hw_config_names = [HWConfigOpName.DIVIDE]


@TF_OPERATION_METATYPES.register()
class TFLogOpMetatype(TFOpMetatype):
    name = 'LogOp'
    op_names = ['Log', 'math.log']


@TF_OPERATION_METATYPES.register()
class TFFloorOpMetatype(TFOpMetatype):
    name = 'FloorOp'
    op_names = ['Floor', 'math.floor']


@TF_OPERATION_METATYPES.register()
class TFFloorDivOpMetatype(TFOpMetatype):
    name = 'FloorDivOp'
    op_names = ['FloorDiv', 'math.floordiv', 'compat.v1.floor_div']
    hw_config_names = [HWConfigOpName.FLOORMOD]


@TF_OPERATION_METATYPES.register()
class TFCastOpMetatype(TFOpMetatype):
    name = 'CastOp'
    op_names = ['Cast', 'cast']


@TF_OPERATION_METATYPES.register()
class TFMaxOpMetatype(TFOpMetatype):
    name = 'MaxOp'
    op_names = ['Max', 'math.reduce_max']
    hw_config_names = [HWConfigOpName.MAXIMUM]


@TF_OPERATION_METATYPES.register()
class TFTanhOpMetatype(TFOpMetatype):
    name = 'TanhOp'
    op_names = ['Tanh', 'math.tanh']


@TF_OPERATION_METATYPES.register()
class TFSeluOpMetatype(TFOpMetatype):
    name = 'SeluOp'
    op_names = ['Selu']


@TF_OPERATION_METATYPES.register()
class TFEluOpMetatype(TFOpMetatype):
    name = 'EluOp'
    op_names = ['Elu']


@TF_OPERATION_METATYPES.register()
class TFLeakyReluOpMetatype(TFOpMetatype):
    name = 'LeakyReluOp'
    op_names = ['LeakyRelu']


@TF_OPERATION_METATYPES.register()
class TFMaxPoolOpMetatype(TFOpMetatype):
    name = 'MaxPoolOp'
    op_names = ['MaxPool']
    hw_config_names = [HWConfigOpName.MAXPOOL]


@TF_OPERATION_METATYPES.register()
class TFMaxPool3DOpMetatype(TFOpMetatype):
    name = 'MaxPool3DOp'
    op_names = ['MaxPool3D']
    hw_config_names = [HWConfigOpName.MAXPOOL]


@TF_OPERATION_METATYPES.register()
class TFNegOpMetatype(TFOpMetatype):
    name = 'NegOp'
    op_names = ['Neg', 'math.negative']


@TF_OPERATION_METATYPES.register()
class TFTileOpMetatype(TFOpMetatype):
    name = 'TileOp'
    op_names = ['Tile', 'tile']
    hw_config_names = [HWConfigOpName.TILE]


@TF_OPERATION_METATYPES.register()
class TFSliceOpMetatype(TFOpMetatype):
    name = 'SliceOp'
    op_names = ['Slice', 'slice']


@TF_OPERATION_METATYPES.register()
class TFSoftmaxOpMetatype(TFOpMetatype):
    name = 'SoftmaxOp'
    op_names = ['Softmax']


@TF_OPERATION_METATYPES.register()
class TFTransposeOpMetatype(TFOpMetatype):
    name = 'TransposeOp'
    op_names = ['Transpose', 'transpose', 'compat.v1.transpose']
    hw_config_names = [HWConfigOpName.TRANSPOSE]


@TF_OPERATION_METATYPES.register()
class TFGreaterOpMetatype(TFOpMetatype):
    name = 'GreaterOp'
    op_names = ['Greater', 'math.greater']
    hw_config_names = [HWConfigOpName.GREATER]


@TF_OPERATION_METATYPES.register()
class TFResizeNearestNeighborOpMetatype(TFOpMetatype):
    name = 'ResizeNearestNeighborOp'
    op_names = ['ResizeNearestNeighbor']
    hw_config_names = [HWConfigOpName.INTERPOLATE]


WEIGHTABLE_TF_OP_METATYPES = [
    TFConv2DOpMetatype,
    TFConv3DOpMetatype,
    TFDepthwiseConv2dNativeOpMetatype,
    TFQuantizedConv2DOpMetatype,
]
