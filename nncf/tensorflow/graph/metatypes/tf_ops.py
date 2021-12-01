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


@TF_OPERATION_METATYPES.register()
class TFNoopMetatype(OperatorMetatype):
    name = 'noop'

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return [cls.name]


@TF_OPERATION_METATYPES.register()
class TFIdentityOpMetatype(TFOpMetatype):
    name = 'IdentityOp'
    op_names = ['Identity']


@TF_OPERATION_METATYPES.register()
class TFPackOpMetatype(TFOpMetatype):
    # Unsqueezes->Concat pattern
    name = 'PackOp'
    op_names = ['Pack']


@TF_OPERATION_METATYPES.register()
class TFPadOpMetatype(TFOpMetatype):
    name = 'PadOp'
    op_names = ['Pad']
    hw_config_names = [HWConfigOpName.PAD]


@TF_OPERATION_METATYPES.register()
class TFStridedSliceOpMetatype(TFOpMetatype):
    name = 'StridedSliceOp'
    op_names = ['StridedSlice']
    hw_config_names = [HWConfigOpName.STRIDEDSLICE]


@TF_OPERATION_METATYPES.register()
class TFConcatOpMetatype(TFOpMetatype):
    name = 'ConcatOp'
    op_names = ['Concat', 'ConcatV2']
    hw_config_names = [HWConfigOpName.CONCAT]


@TF_OPERATION_METATYPES.register()
class TFAddOpMetatype(TFOpMetatype):
    name = 'AddOp'
    op_names = ['Add', 'AddV2']
    hw_config_names = [HWConfigOpName.ADD]


@TF_OPERATION_METATYPES.register()
class TFSubOpMetatype(TFOpMetatype):
    name = 'SubOp'
    op_names = ['Sub']
    hw_config_names = [HWConfigOpName.SUBTRACT]


@TF_OPERATION_METATYPES.register()
class TFMulOpMetatype(TFOpMetatype):
    name = 'MulOp'
    op_names = ['Mul']
    hw_config_names = [HWConfigOpName.MULTIPLY]


@TF_OPERATION_METATYPES.register()
class TFAvgPoolOpMetatype(TFOpMetatype):
    name = 'AvgPoolOp'
    op_names = ['AvgPool']
    hw_config_names = [HWConfigOpName.AVGPOOL]


@TF_OPERATION_METATYPES.register()
class TFAvgPool3DOpMetatype(TFOpMetatype):
    name = 'AvgPool3DOp'
    op_names = ['AvgPool3D']
    hw_config_names = [HWConfigOpName.AVGPOOL]


@TF_OPERATION_METATYPES.register()
class TFReluOpMetatype(TFOpMetatype):
    name = 'ReluOp'
    op_names = ['Relu']


@TF_OPERATION_METATYPES.register()
class TFRelu6OpMetatype(TFOpMetatype):
    name = 'Relu6Op'
    op_names = ['Relu6']


@TF_OPERATION_METATYPES.register()
class TFMatMulOpMetatype(TFOpMetatype):
    name = 'MatMulOp'
    op_names = ['MatMul']


@TF_OPERATION_METATYPES.register()
class TFConv2DOpMetatype(TFOpMetatype):
    name = 'Conv2DOp'
    op_names = ['Conv2D']


@TF_OPERATION_METATYPES.register()
class TFConv3DOpMetatype(TFOpMetatype):
    name = 'Conv3DOp'
    op_names = ['Conv3D']


@TF_OPERATION_METATYPES.register()
class TFDepthwiseConv2dNativeOpMetatype(TFOpMetatype):
    name = 'DepthwiseConv2dNativeOp'
    op_names = ['DepthwiseConv2dNative']


@TF_OPERATION_METATYPES.register()
class TFQuantizedConv2DOpMetatype(TFOpMetatype):
    name = 'QuantizedConv2DOp'
    op_names = ['QuantizedConv2D']


@TF_OPERATION_METATYPES.register()
class TFReshapeOpMetatype(TFOpMetatype):
    name = 'ReshapeOp'
    op_names = ['Reshape']
    hw_config_names = [HWConfigOpName.RESHAPE]


@TF_OPERATION_METATYPES.register()
class TFExpandDimsOpMetatype(TFOpMetatype):
    name = 'ExpandDimsOp'
    op_names = ['ExpandDims']


@TF_OPERATION_METATYPES.register()
class TFSplitOpMetatype(TFOpMetatype):
    name = 'SplitOp'
    op_names = ['Split']
    hw_config_names = [HWConfigOpName.SPLIT]


@TF_OPERATION_METATYPES.register()
class TFMinimumOpMetatype(TFOpMetatype):
    name = 'MinimumOp'
    op_names = ['Minimum']
    hw_config_names = [HWConfigOpName.MINIMUM]


@TF_OPERATION_METATYPES.register()
class TFMaximumOpMetatype(TFOpMetatype):
    name = 'MaximumOp'
    op_names = ['Maximum', 'Max']
    hw_config_names = [HWConfigOpName.MAXIMUM]


@TF_OPERATION_METATYPES.register()
class TFExpOpMetatype(TFOpMetatype):
    name = 'ExpOp'
    op_names = ['Exp']


class TFPlaceholderOpMetatype(TFOpMetatype):
    name = 'PlaceholderOp'
    op_names = ['Placeholder']


@TF_OPERATION_METATYPES.register()
class TFShapeOpMetatype(TFOpMetatype):
    name = 'ShapeOp'
    op_names = ['Shape']


@TF_OPERATION_METATYPES.register()
class TFBiasAddOpMetatype(TFOpMetatype):
    name = 'BiasAddOp'
    op_names = ['BiasAdd']
    hw_config_names = [HWConfigOpName.ADD]


@TF_OPERATION_METATYPES.register()
class TFMeanOpMetatype(TFOpMetatype):
    name = 'MeanOp'
    op_names = ['Mean']
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
    op_names = ['Squeeze']
    hw_config_names = [HWConfigOpName.SQUEEZE]


@TF_OPERATION_METATYPES.register()
class TFSigmoidOpMetatype(TFOpMetatype):
    name = 'SigmoidOp'
    op_names = ['Sigmoid']


@TF_OPERATION_METATYPES.register()
class TFCombinedNonMaxSuppressionOpMetatype(TFOpMetatype):
    name = 'CombinedNonMaxSuppressionOp'
    op_names = ['CombinedNonMaxSuppression']


@TF_OPERATION_METATYPES.register()
class TFTopKV2OpMetatype(TFOpMetatype):
    name = 'TopKV2Op'
    op_names = ['TopKV2']


@TF_OPERATION_METATYPES.register()
class TFGatherOpMetatype(TFOpMetatype):
    name = 'GatherOp'
    op_names = ['GatherNd', 'GatherV2']


@TF_OPERATION_METATYPES.register()
class TFPowOpMetatype(TFOpMetatype):
    name = 'PowOp'
    op_names = ['Pow', 'Sqrt']
    hw_config_names = [HWConfigOpName.POWER]


@TF_OPERATION_METATYPES.register()
class TFTrueDivOpMetatype(TFOpMetatype):
    name = 'TrueDivOp'
    op_names = ['RealDiv']
    hw_config_names = [HWConfigOpName.DIVIDE]


@TF_OPERATION_METATYPES.register()
class TFLogOpMetatype(TFOpMetatype):
    name = 'LogOp'
    op_names = ['Log']


@TF_OPERATION_METATYPES.register()
class TFFloorOpMetatype(TFOpMetatype):
    name = 'FloorOp'
    op_names = ['Floor']


@TF_OPERATION_METATYPES.register()
class TFFloorDivOpMetatype(TFOpMetatype):
    name = 'FloorDivOp'
    op_names = ['FloorDiv']


@TF_OPERATION_METATYPES.register()
class TFCastOpMetatype(TFOpMetatype):
    name = 'CastOp'
    op_names = ['Cast']


class TFMaxOpMetatype(TFOpMetatype):
    name = 'MaxOp'
    op_names = ['Max']
    hw_config_names = [HWConfigOpName.MAXIMUM]


@TF_OPERATION_METATYPES.register()
class TFTanhOpMetatype(TFOpMetatype):
    name = 'TanhOp'
    op_names = ['Tanh']


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
    op_names = ['Neg']


@TF_OPERATION_METATYPES.register()
class TFTileOpMetatype(TFOpMetatype):
    name = 'TileOp'
    op_names = ['Tile']
    hw_config_names = [HWConfigOpName.TILE]


@TF_OPERATION_METATYPES.register()
class TFSliceOpMetatype(TFOpMetatype):
    name = 'SliceOp'
    op_names = ['Slice']


@TF_OPERATION_METATYPES.register()
class TFSoftmaxOpMetatype(TFOpMetatype):
    name = 'SoftmaxOp'
    op_names = ['Softmax']


@TF_OPERATION_METATYPES.register()
class TFTransposeOpMetatype(TFOpMetatype):
    name = 'TransposeOp'
    op_names = ['Transpose']
    hw_config_names = [HWConfigOpName.TRANSPOSE]


class TFGreaterOpMetatype(TFOpMetatype):
    name = 'GreaterOp'
    op_names = ['Greater']
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
