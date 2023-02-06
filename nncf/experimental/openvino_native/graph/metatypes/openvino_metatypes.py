"""
 Copyright (c) 2023 Intel Corporation
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

from typing import List, Optional, Type
import openvino.runtime as ov

from nncf.common.graph.operator_metatypes import INPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OUTPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatypeRegistry
from nncf.common.hardware.opset import HWConfigOpName

OV_OPERATOR_METATYPES = OperatorMetatypeRegistry('openvino_operator_metatypes')


class OVOpMetatype(OperatorMetatype):
    op_names = []  # type: List[str]
    subtypes = []  # type: List[Type[OperatorMetatype]]

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return cls.op_names

    @classmethod
    def get_subtypes(cls) -> List[Type[OperatorMetatype]]:
        return cls.subtypes

    @classmethod
    def matches(cls, node: ov.Node) -> Optional[bool]:
        return node.op_type in cls.op_names

    @classmethod
    def determine_subtype(cls, node: ov.Node) -> Optional[Type[OperatorMetatype]]:
        matches = []
        for subtype in cls.get_subtypes():
            if subtype.matches(node):
                matches.append(subtype)
        if len(matches) > 1:
            raise RuntimeError('Multiple subtypes match operator call - '
                               'can not determine single subtype.')
        if not matches:
            return None
        return matches[0]


@OV_OPERATOR_METATYPES.register()
class OVConvolutionMetatype(OVOpMetatype):
    name = 'ConvOp'
    op_names = ['Convolution']
    hw_config_names = [HWConfigOpName.CONVOLUTION]


@OV_OPERATOR_METATYPES.register()
class OVConvolutionBackpropDataMetatype(OVOpMetatype):
    name = 'ConvBackpropDataOp'
    op_names = ['ConvolutionBackpropData']
    hw_config_names = [HWConfigOpName.CONVOLUTION]


@OV_OPERATOR_METATYPES.register()
class OVDepthwiseConvolutionMetatype(OVOpMetatype):
    name = 'DepthwiseConvolutionOp'
    op_names = ['GroupConvolution']
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]

    @classmethod
    def matches(cls, node: ov.Node) -> bool:
        return _is_depthwise_conv(node)


@OV_OPERATOR_METATYPES.register()
class OVGroupConvolutionMetatype(OVOpMetatype):
    name = 'GroupConvolutionOp'
    op_names = ['GroupConvolution']
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    subtypes = [OVDepthwiseConvolutionMetatype]


@OV_OPERATOR_METATYPES.register()
class OVGroupConvolutionBackpropDataMetatype(OVOpMetatype):
    name = 'GroupConvolutionBackpropDataOp'
    op_names = ['GroupConvolutionBackpropData']
    hw_config_names = [HWConfigOpName.CONVOLUTION]


@OV_OPERATOR_METATYPES.register()
class OVMatMulMetatype(OVOpMetatype):
    name = 'MatMulOp'
    op_names = ['MatMul']
    hw_config_names = [HWConfigOpName.MATMUL]


@OV_OPERATOR_METATYPES.register()
class OVReluMetatype(OVOpMetatype):
    name = 'ReluOp'
    op_names = ['Relu']


@OV_OPERATOR_METATYPES.register()
class OVGeluMetatype(OVOpMetatype):
    name = 'GeluOp'
    op_names = ['Gelu']
    hw_config_names = [HWConfigOpName.GELU]


@OV_OPERATOR_METATYPES.register()
class OVEluMetatype(OVOpMetatype):
    name = 'EluOp'
    op_names = ['Elu']


@OV_OPERATOR_METATYPES.register()
class OVPReluMetatype(OVOpMetatype):
    name = 'PReluOp'
    op_names = ['PReLU']


@OV_OPERATOR_METATYPES.register()
class OVSigmoidMetatype(OVOpMetatype):
    name = 'SigmoidOp'
    op_names = ['Sigmoid']


@OV_OPERATOR_METATYPES.register()
class OVHardSigmoidMetatype(OVOpMetatype):
    name = 'HardSigmoidOp'
    op_names = ['HardSigmoid']


@OV_OPERATOR_METATYPES.register()
class OVAvgPoolMetatype(OVOpMetatype):
    name = 'AvgPoolOp'
    op_names = ['AvgPool']
    hw_config_names = [HWConfigOpName.AVGPOOL]


@OV_OPERATOR_METATYPES.register()
class OVMaxPoolMetatype(OVOpMetatype):
    name = 'MaxPoolOp'
    op_names = ['MaxPool']
    hw_config_names = [HWConfigOpName.MAXPOOL]


@OV_OPERATOR_METATYPES.register()
class OVConstantMetatype(OVOpMetatype):
    name = 'ConstantOp'
    op_names = ['Constant']


@OV_OPERATOR_METATYPES.register()
class OVAddMetatype(OVOpMetatype):
    name = 'AddOp'
    op_names = ['Add']
    hw_config_names = [HWConfigOpName.ADD]


@OV_OPERATOR_METATYPES.register()
class OVSubtractMetatype(OVOpMetatype):
    name = 'SubtractOp'
    op_names = ['Subtract']
    hw_config_names = [HWConfigOpName.SUBTRACT]


@OV_OPERATOR_METATYPES.register()
class OVMultiplyMetatype(OVOpMetatype):
    name = 'MultiplyOp'
    op_names = ['Multiply']
    hw_config_names = [HWConfigOpName.MULTIPLY]


@OV_OPERATOR_METATYPES.register()
class OVDivideMetatype(OVOpMetatype):
    name = 'DivideOp'
    op_names = ['Divide']
    hw_config_names = [HWConfigOpName.DIVIDE]


@OV_OPERATOR_METATYPES.register()
class OVSumMetatype(OVOpMetatype):
    name = 'SumOp'
    op_names = ['ReduceSum']
    hw_config_names = [HWConfigOpName.REDUCESUM]


@OV_OPERATOR_METATYPES.register()
class OVConcatMetatype(OVOpMetatype):
    name = 'ConcatOp'
    op_names = ['Concat']
    hw_config_names = [HWConfigOpName.CONCAT]


@OV_OPERATOR_METATYPES.register()
class OVBatchNormMetatype(OVOpMetatype):
    name = 'BatchNormalizationOp'
    op_names = ['BatchNormInference']


@OV_OPERATOR_METATYPES.register()
class OVInterpolateMetatype(OVOpMetatype):
    name = 'InterpolateOp'
    op_names = ['Interpolate']
    hw_config_names = [HWConfigOpName.INTERPOLATE]


@OV_OPERATOR_METATYPES.register()
class OVMVNMetatype(OVOpMetatype):
    name = 'MVNOp'
    op_names = ['MVN']
    hw_config_names = [HWConfigOpName.MVN]


@OV_OPERATOR_METATYPES.register()
class OVNormalizeL2Metatype(OVOpMetatype):
    name = 'NormalizeL2Op'
    op_names = ['NormalizeL2']


@OV_OPERATOR_METATYPES.register()
class OVReshapeMetatype(OVOpMetatype):
    name = 'ReshapeOp'
    op_names = ['Reshape']
    hw_config_names = [HWConfigOpName.RESHAPE]


@OV_OPERATOR_METATYPES.register()
class OVShapeMetatype(OVOpMetatype):
    name = 'ShapeOp'
    op_names = ['ShapeOf']


@OV_OPERATOR_METATYPES.register()
class OVNonZeroMetatype(OVOpMetatype):
    name = 'NonZeroOp'
    op_names = ['NonZero']


@OV_OPERATOR_METATYPES.register()
class OVSplitMetatype(OVOpMetatype):
    name = 'SplitOp'
    op_names = ['Split']
    hw_config_names = [HWConfigOpName.SPLIT]


@OV_OPERATOR_METATYPES.register()
class OVVariadicSplitMetatype(OVOpMetatype):
    name = 'VariadicSplitOp'
    op_names = ['VariadicSplit']


@OV_OPERATOR_METATYPES.register()
class OVShuffleChannelsMetatype(OVOpMetatype):
    name = 'ShuffleChannelsOp'
    op_names = ['ShuffleChannels']


@OV_OPERATOR_METATYPES.register()
class OVBroadcastMetatype(OVOpMetatype):
    name = 'BroadcastOp'
    op_names = ['Broadcast']


@OV_OPERATOR_METATYPES.register()
class OVConvertLikeMetatype(OVOpMetatype):
    name = 'ConvertLikeOp'
    op_names = ['ConvertLike']


@OV_OPERATOR_METATYPES.register()
class OVDepthToSpaceMetatype(OVOpMetatype):
    name = 'DepthToSpaceOp'
    op_names = ['DepthToSpace']


@OV_OPERATOR_METATYPES.register()
class OVLSTMSequenceMetatype(OVOpMetatype):
    name = 'LSTMSequenceOp'
    op_names = ['LSTMSequence']


@OV_OPERATOR_METATYPES.register()
class OVGRUSequenceMetatype(OVOpMetatype):
    name = 'GRUSequenceOp'
    op_names = ['GRUSequence']


@OV_OPERATOR_METATYPES.register()
class OVFakeQuantizeMetatype(OVOpMetatype):
    name = 'FakeQuantizeOp'
    op_names = ['FakeQuantize']


@OV_OPERATOR_METATYPES.register()
class OVLessMetatype(OVOpMetatype):
    name = 'LessOp'
    op_names = ['Less']
    hw_config_names = [HWConfigOpName.LESS]


@OV_OPERATOR_METATYPES.register()
class OVLessEqualMetatype(OVOpMetatype):
    name = 'LessEqualOp'
    op_names = ['LessEqual']
    hw_config_names = [HWConfigOpName.LESSEQUAL]


@OV_OPERATOR_METATYPES.register()
class OVGreaterMetatype(OVOpMetatype):
    name = 'GreaterOp'
    op_names = ['Greater']
    hw_config_names = [HWConfigOpName.GREATER]


@OV_OPERATOR_METATYPES.register()
class OVGreaterEqualMetatype(OVOpMetatype):
    name = 'GreaterEqualOp'
    op_names = ['GreaterEqual']
    hw_config_names = [HWConfigOpName.GREATEREQUAL]


@OV_OPERATOR_METATYPES.register()
class OVEqualMetatype(OVOpMetatype):
    name = 'EqualOp'
    op_names = ['Equal']
    hw_config_names = [HWConfigOpName.EQUAL]


@OV_OPERATOR_METATYPES.register()
class OVNotEqualMetatype(OVOpMetatype):
    name = 'NotEqualOp'
    op_names = ['NotEqual']
    hw_config_names = [HWConfigOpName.NOTEQUAL]


@OV_OPERATOR_METATYPES.register()
class OVLogicalNotMetatype(OVOpMetatype):
    name = 'LogicalNotOp'
    op_names = ['LogicalNot']
    hw_config_names = [HWConfigOpName.LOGICALNOT]


@OV_OPERATOR_METATYPES.register()
class OVLogicalAndMetatype(OVOpMetatype):
    name = 'LogicalAndOp'
    op_names = ['LogicalAnd']
    hw_config_names = [HWConfigOpName.LOGICALAND]


@OV_OPERATOR_METATYPES.register()
class OVLogicalOrMetatype(OVOpMetatype):
    name = 'LogicalOrOp'
    op_names = ['LogicalOr']
    hw_config_names = [HWConfigOpName.LOGICALOR]


@OV_OPERATOR_METATYPES.register()
class OVLogicalXorMetatype(OVOpMetatype):
    name = 'LogicalXorOp'
    op_names = ['LogicalXor']
    hw_config_names = [HWConfigOpName.LOGICALXOR]


@OV_OPERATOR_METATYPES.register()
class OVFloorMetatype(OVOpMetatype):
    name = 'FloorOp'
    op_names = ['Floor']


@OV_OPERATOR_METATYPES.register()
class OVFloorModMetatype(OVOpMetatype):
    name = 'FloorModOp'
    op_names = ['FloorMod']
    hw_config_names = [HWConfigOpName.FLOORMOD]


@OV_OPERATOR_METATYPES.register()
class OVMaximumMetatype(OVOpMetatype):
    name = 'MaximumOp'
    op_names = ['Maximum']
    hw_config_names = [HWConfigOpName.MAXIMUM]

@OV_OPERATOR_METATYPES.register()
class OVMinimumMetatype(OVOpMetatype):
    name = 'MinimumOp'
    op_names = ['Minimum']
    hw_config_names = [HWConfigOpName.MINIMUM]


@OV_OPERATOR_METATYPES.register()
class OVSqrtMetatype(OVOpMetatype):
    name = 'SqrtOp'
    op_names = ['Sqrt']
    hw_config_names = [HWConfigOpName.POWER]


@OV_OPERATOR_METATYPES.register()
class OVPowerMetatype(OVOpMetatype):
    name = 'PowerOp'
    op_names = ['Power']
    hw_config_names = [HWConfigOpName.POWER]


@OV_OPERATOR_METATYPES.register()
class OVLogMetatype(OVOpMetatype):
    name = 'LogOp'
    op_names = ['Log']


@OV_OPERATOR_METATYPES.register()
class OVRoiAlignMetatype(OVOpMetatype):
    name = 'RoiAlignOp'
    op_names = ['ROIAlign']


@OV_OPERATOR_METATYPES.register()
class OVGatherMetatype(OVOpMetatype):
    name = 'GatherOp'
    op_names = ['Gather']


@OV_OPERATOR_METATYPES.register()
class OVUnsqueezeMetatype(OVOpMetatype):
    name = 'UnsqueezeOp'
    op_names = ['Unsqueeze']
    hw_config_names = [HWConfigOpName.UNSQUEEZE]


@OV_OPERATOR_METATYPES.register()
class OVSqueezeMetatype(OVOpMetatype):
    name = 'SqueezeOp'
    op_names = ['Squeeze']
    hw_config_names = [HWConfigOpName.SQUEEZE]


@OV_OPERATOR_METATYPES.register()
class OVNonMaxSuppressionMetatype(OVOpMetatype):
    name = 'NonMaxSuppressionOp'
    op_names = ['NonMaxSuppression']


@OV_OPERATOR_METATYPES.register()
class OVReduceMinMetatype(OVOpMetatype):
    name = 'ReduceMinOp'
    op_names = ['ReduceMin']


@OV_OPERATOR_METATYPES.register()
class OVReduceMaxMetatype(OVOpMetatype):
    name = 'ReduceMaxOp'
    op_names = ['ReduceMax']
    hw_config_names = [HWConfigOpName.REDUCEMAX]


@OV_OPERATOR_METATYPES.register()
class OVReduceMeanMetatype(OVOpMetatype):
    name = 'ReduceMeanOp'
    op_names = ['ReduceMean']
    hw_config_names = [HWConfigOpName.REDUCEMEAN]


@OV_OPERATOR_METATYPES.register()
class OVReduceL1Metatype(OVOpMetatype):
    name = 'ReduceL1Op'
    op_names = ['ReduceL1']


@OV_OPERATOR_METATYPES.register()
class OVReduceL2Metatype(OVOpMetatype):
    name = 'ReduceL2Op'
    op_names = ['ReduceL2']
    hw_config_names = [HWConfigOpName.REDUCEL2]


@OV_OPERATOR_METATYPES.register()
class OVTopKMetatype(OVOpMetatype):
    name = 'TopKOp'
    op_names = ['TopK']


@OV_OPERATOR_METATYPES.register()
class OVStridedSliceMetatype(OVOpMetatype):
    name = 'StridedSliceOp'
    op_names = ['StridedSlice']
    hw_config_names = [HWConfigOpName.STRIDEDSLICE]


@OV_OPERATOR_METATYPES.register()
class OVExpMetatype(OVOpMetatype):
    name = 'ExpOp'
    op_names = ['Exp']


@OV_OPERATOR_METATYPES.register()
class OVTransposeMetatype(OVOpMetatype):
    name = 'TransposeOp'
    op_names = ['Transpose']
    hw_config_names = [HWConfigOpName.TRANSPOSE]


@OV_OPERATOR_METATYPES.register()
class OVTileMetatype(OVOpMetatype):
    name = 'TileOp'
    op_names = ['Tile']
    hw_config_names = [HWConfigOpName.TILE]


@OV_OPERATOR_METATYPES.register()
class OVSoftmaxMetatype(OVOpMetatype):
    name = 'SoftmaxOp'
    op_names = ['SoftMax']


@OV_OPERATOR_METATYPES.register()
class OVPadMetatype(OVOpMetatype):
    name = 'PadOp'
    op_names = ['Pad']
    hw_config_names = [HWConfigOpName.PAD]


@OV_OPERATOR_METATYPES.register()
class OVReadValueMetatype(OVOpMetatype):
    name = 'ReadValueOp'
    op_names = ['ReadValue']


@OV_OPERATOR_METATYPES.register()
class OVAssignMetatype(OVOpMetatype):
    name = 'AssignOp'
    op_names = ['Assign']


@OV_OPERATOR_METATYPES.register()
class OVConvertMetatype(OVOpMetatype):
    name = 'ConvertOp'
    op_names = ['Convert']


@OV_OPERATOR_METATYPES.register()
@INPUT_NOOP_METATYPES.register()
class OVParameterMetatype(OVOpMetatype):
    name = 'ParameterOp'
    op_names = ['Parameter']


@OV_OPERATOR_METATYPES.register()
@OUTPUT_NOOP_METATYPES.register()
class OVResultMetatype(OVOpMetatype):
    name = 'ResultOp'
    op_names = ['Result']


GENERAL_WEIGHT_LAYER_METATYPES = [OVConvolutionMetatype,
                                  OVGroupConvolutionMetatype,
                                  OVDepthwiseConvolutionMetatype,
                                  OVConvolutionBackpropDataMetatype,
                                  OVGroupConvolutionBackpropDataMetatype,
                                  OVMatMulMetatype]

METATYPES_WITH_CONST_PORT_ID = GENERAL_WEIGHT_LAYER_METATYPES + [OVAddMetatype]

# Contains the operation metatypes for which bias can be applied.
OPERATIONS_WITH_BIAS_METATYPES = [OVConvolutionMetatype,
                                  OVConvolutionBackpropDataMetatype,
                                  OVMatMulMetatype]


def get_operator_metatypes() -> List[Type[OperatorMetatype]]:
    """
    Returns a list of the operator metatypes.
    :return: List of operator metatypes .
    """
    return list(OV_OPERATOR_METATYPES.registry_dict.values())


def _is_depthwise_conv(node: ov.Node) -> bool:
    """
    Returns True if the group convolution is depthwise, False - otherwise.
    Depthwise convolution is a convolution satisfies the following rule:
    groups == in_channels and inp_channels > 1.
    Weight tensor layout is [groups, output channels / groups, input channels / groups, Z, Y, X],
    where Z, Y, X - spatial axes.

    :param node: GroupConvolution node to check whether it is depthwise.
    :return: True if the convolution is depthwise, False - otherwise.
    """
    inp_channels = node.input_value(0).get_shape()[1]
    groups = node.input_value(1).get_shape()[0]
    return groups == inp_channels and inp_channels > 1
