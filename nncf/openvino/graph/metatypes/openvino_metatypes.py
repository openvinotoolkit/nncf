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

from collections import deque
from typing import List, Optional, Type

import openvino.runtime as ov

import nncf
from nncf.common.graph.operator_metatypes import INPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OUTPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatypeRegistry
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.hardware.opset import HWConfigOpName

OV_OPERATOR_METATYPES = OperatorMetatypeRegistry("openvino_operator_metatypes")


class OVOpMetatype(OperatorMetatype):
    op_names: List[str] = []
    subtypes: List[Type[OperatorMetatype]] = []

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
            raise nncf.InternalError("Multiple subtypes match operator call - can not determine single subtype.")
        if not matches:
            return None
        return matches[0]


@OV_OPERATOR_METATYPES.register()
class OVConvolutionMetatype(OVOpMetatype):
    name = "ConvOp"
    op_names = ["Convolution"]
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    output_channel_axis = 1


@OV_OPERATOR_METATYPES.register()
class OVConvolutionBackpropDataMetatype(OVOpMetatype):
    name = "ConvBackpropDataOp"
    op_names = ["ConvolutionBackpropData"]
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    output_channel_axis = 1


@OV_OPERATOR_METATYPES.register(is_subtype=True)
class OVDepthwiseConvolutionMetatype(OVOpMetatype):
    name = "DepthwiseConvolutionOp"
    op_names = ["GroupConvolution"]
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]
    output_channel_axis = 1

    @classmethod
    def matches(cls, node: ov.Node) -> bool:
        return _is_depthwise_conv(node)


@OV_OPERATOR_METATYPES.register()
class OVGroupConvolutionMetatype(OVOpMetatype):
    name = "GroupConvolutionOp"
    op_names = ["GroupConvolution"]
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    subtypes = [OVDepthwiseConvolutionMetatype]
    output_channel_axis = 1


@OV_OPERATOR_METATYPES.register()
class OVGroupConvolutionBackpropDataMetatype(OVOpMetatype):
    name = "GroupConvolutionBackpropDataOp"
    op_names = ["GroupConvolutionBackpropData"]
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    output_channel_axis = 1


@OV_OPERATOR_METATYPES.register()
class OVMatMulMetatype(OVOpMetatype):
    name = "MatMulOp"
    op_names = ["MatMul"]
    hw_config_names = [HWConfigOpName.MATMUL]
    output_channel_axis = -1


@OV_OPERATOR_METATYPES.register()
class OVReluMetatype(OVOpMetatype):
    name = "ReluOp"
    op_names = ["Relu"]


@OV_OPERATOR_METATYPES.register()
class OVGeluMetatype(OVOpMetatype):
    name = "GeluOp"
    op_names = ["Gelu"]
    hw_config_names = [HWConfigOpName.GELU]


@OV_OPERATOR_METATYPES.register()
class OVEluMetatype(OVOpMetatype):
    name = "EluOp"
    op_names = ["Elu"]


@OV_OPERATOR_METATYPES.register()
class OVPReluMetatype(OVOpMetatype):
    name = "PReluOp"
    op_names = ["PRelu"]


@OV_OPERATOR_METATYPES.register()
class OVSigmoidMetatype(OVOpMetatype):
    name = "SigmoidOp"
    op_names = ["Sigmoid"]


@OV_OPERATOR_METATYPES.register()
class OVHSigmoidMetatype(OVOpMetatype):
    name = "HSigmoidOp"
    op_names = ["HSigmoid"]


@OV_OPERATOR_METATYPES.register()
class OVHardSigmoidMetatype(OVOpMetatype):
    name = "HardSigmoidOp"
    op_names = ["HardSigmoid"]


@OV_OPERATOR_METATYPES.register()
class OVAvgPoolMetatype(OVOpMetatype):
    name = "AvgPoolOp"
    op_names = ["AvgPool"]
    hw_config_names = [HWConfigOpName.AVGPOOL]


@OV_OPERATOR_METATYPES.register()
class OVAdaptiveAvgPoolMetatype(OVOpMetatype):
    name = "AdaptiveAvgPoolOp"
    op_names = ["AdaptiveAvgPool"]
    hw_config_names = [HWConfigOpName.AVGPOOL]


@OV_OPERATOR_METATYPES.register()
class OVMaxPoolMetatype(OVOpMetatype):
    name = "MaxPoolOp"
    op_names = ["MaxPool"]
    hw_config_names = [HWConfigOpName.MAXPOOL]


@OV_OPERATOR_METATYPES.register()
class OVAdaptiveMaxPoolMetatype(OVOpMetatype):
    name = "AdaptiveMaxPoolOp"
    op_names = ["AdaptiveMaxPool"]
    hw_config_names = [HWConfigOpName.MAXPOOL]


@OV_OPERATOR_METATYPES.register()
class OVConstantMetatype(OVOpMetatype):
    name = "ConstantOp"
    op_names = ["Constant"]


@OV_OPERATOR_METATYPES.register()
class OVAddMetatype(OVOpMetatype):
    name = "AddOp"
    op_names = ["Add"]
    hw_config_names = [HWConfigOpName.ADD]


@OV_OPERATOR_METATYPES.register()
class OVSubtractMetatype(OVOpMetatype):
    name = "SubtractOp"
    op_names = ["Subtract"]
    hw_config_names = [HWConfigOpName.SUBTRACT]


@OV_OPERATOR_METATYPES.register()
class OVMultiplyMetatype(OVOpMetatype):
    name = "MultiplyOp"
    op_names = ["Multiply"]
    hw_config_names = [HWConfigOpName.MULTIPLY]


@OV_OPERATOR_METATYPES.register()
class OVDivideMetatype(OVOpMetatype):
    name = "DivideOp"
    op_names = ["Divide"]
    hw_config_names = [HWConfigOpName.DIVIDE]


@OV_OPERATOR_METATYPES.register()
class OVSumMetatype(OVOpMetatype):
    name = "SumOp"
    op_names = ["ReduceSum"]
    hw_config_names = [HWConfigOpName.REDUCESUM]


@OV_OPERATOR_METATYPES.register()
class OVConcatMetatype(OVOpMetatype):
    name = "ConcatOp"
    op_names = ["Concat"]
    hw_config_names = [HWConfigOpName.CONCAT]


@OV_OPERATOR_METATYPES.register()
class OVBatchNormMetatype(OVOpMetatype):
    name = "BatchNormalizationOp"
    op_names = ["BatchNormInference"]


@OV_OPERATOR_METATYPES.register()
class OVInterpolateMetatype(OVOpMetatype):
    name = "InterpolateOp"
    op_names = ["Interpolate"]
    hw_config_names = [HWConfigOpName.INTERPOLATE]


@OV_OPERATOR_METATYPES.register()
class OVMVNMetatype(OVOpMetatype):
    name = "MVNOp"
    op_names = ["MVN"]
    hw_config_names = [HWConfigOpName.MVN]


@OV_OPERATOR_METATYPES.register()
class OVNormalizeL2Metatype(OVOpMetatype):
    name = "NormalizeL2Op"
    op_names = ["NormalizeL2"]


@OV_OPERATOR_METATYPES.register()
class OVReshapeMetatype(OVOpMetatype):
    name = "ReshapeOp"
    op_names = ["Reshape"]
    hw_config_names = [HWConfigOpName.RESHAPE]


@OV_OPERATOR_METATYPES.register()
class OVShapeOfMetatype(OVOpMetatype):
    name = "ShapeOfOp"
    op_names = ["ShapeOf"]


@OV_OPERATOR_METATYPES.register()
class OVNonZeroMetatype(OVOpMetatype):
    name = "NonZeroOp"
    op_names = ["NonZero"]


@OV_OPERATOR_METATYPES.register()
class OVSplitMetatype(OVOpMetatype):
    name = "SplitOp"
    op_names = ["Split"]
    hw_config_names = [HWConfigOpName.SPLIT]


@OV_OPERATOR_METATYPES.register()
class OVVariadicSplitMetatype(OVOpMetatype):
    name = "VariadicSplitOp"
    op_names = ["VariadicSplit"]


@OV_OPERATOR_METATYPES.register()
class OVShuffleChannelsMetatype(OVOpMetatype):
    name = "ShuffleChannelsOp"
    op_names = ["ShuffleChannels"]


@OV_OPERATOR_METATYPES.register()
class OVBroadcastMetatype(OVOpMetatype):
    name = "BroadcastOp"
    op_names = ["Broadcast"]


@OV_OPERATOR_METATYPES.register()
class OVConvertLikeMetatype(OVOpMetatype):
    name = "ConvertLikeOp"
    op_names = ["ConvertLike"]


@OV_OPERATOR_METATYPES.register()
class OVSpaceToBatchMetatype(OVOpMetatype):
    name = "SpaceToBatchOp"
    op_names = ["SpaceToBatch"]


@OV_OPERATOR_METATYPES.register()
class OVBatchToSpaceMetatype(OVOpMetatype):
    name = "BatchToSpaceOp"
    op_names = ["BatchToSpace"]


@OV_OPERATOR_METATYPES.register()
class OVDepthToSpaceMetatype(OVOpMetatype):
    name = "DepthToSpaceOp"
    op_names = ["DepthToSpace"]


@OV_OPERATOR_METATYPES.register()
class OVSpaceToDepthMetatype(OVOpMetatype):
    name = "SpaceToDepthOp"
    op_names = ["SpaceToDepth"]


@OV_OPERATOR_METATYPES.register()
class OVLSTMSequenceMetatype(OVOpMetatype):
    name = "LSTMSequenceOp"
    op_names = ["LSTMSequence"]
    hw_config_names = [HWConfigOpName.LSTMSEQUENCE]
    const_channel_axis = [1]  # const layout: [num_directions, 4 \* hidden_size, input_size]


@OV_OPERATOR_METATYPES.register()
class OVGRUSequenceMetatype(OVOpMetatype):
    name = "GRUSequenceOp"
    op_names = ["GRUSequence"]
    hw_config_names = [HWConfigOpName.GRUSEQUENCE]
    const_channel_axis = [1]  # const layout: [num_directions, 3 \* hidden_size, input_size]


@OV_OPERATOR_METATYPES.register()
class OVFakeQuantizeMetatype(OVOpMetatype):
    name = "FakeQuantizeOp"
    op_names = ["FakeQuantize"]


@OV_OPERATOR_METATYPES.register()
class OVFakeConvertMetatype(OVOpMetatype):
    name = "FakeConvertOp"
    op_names = ["FakeConvert"]


@OV_OPERATOR_METATYPES.register()
class OVLessMetatype(OVOpMetatype):
    name = "LessOp"
    op_names = ["Less"]
    hw_config_names = [HWConfigOpName.LESS]


@OV_OPERATOR_METATYPES.register()
class OVLessEqualMetatype(OVOpMetatype):
    name = "LessEqualOp"
    op_names = ["LessEqual"]
    hw_config_names = [HWConfigOpName.LESSEQUAL]


@OV_OPERATOR_METATYPES.register()
class OVGreaterMetatype(OVOpMetatype):
    name = "GreaterOp"
    op_names = ["Greater"]
    hw_config_names = [HWConfigOpName.GREATER]


@OV_OPERATOR_METATYPES.register()
class OVGreaterEqualMetatype(OVOpMetatype):
    name = "GreaterEqualOp"
    op_names = ["GreaterEqual"]
    hw_config_names = [HWConfigOpName.GREATEREQUAL]


@OV_OPERATOR_METATYPES.register()
class OVEqualMetatype(OVOpMetatype):
    name = "EqualOp"
    op_names = ["Equal"]
    hw_config_names = [HWConfigOpName.EQUAL]


@OV_OPERATOR_METATYPES.register()
class OVNotEqualMetatype(OVOpMetatype):
    name = "NotEqualOp"
    op_names = ["NotEqual"]
    hw_config_names = [HWConfigOpName.NOTEQUAL]


@OV_OPERATOR_METATYPES.register()
class OVLogicalNotMetatype(OVOpMetatype):
    name = "LogicalNotOp"
    op_names = ["LogicalNot"]
    hw_config_names = [HWConfigOpName.LOGICALNOT]


@OV_OPERATOR_METATYPES.register()
class OVLogicalAndMetatype(OVOpMetatype):
    name = "LogicalAndOp"
    op_names = ["LogicalAnd"]
    hw_config_names = [HWConfigOpName.LOGICALAND]


@OV_OPERATOR_METATYPES.register()
class OVLogicalOrMetatype(OVOpMetatype):
    name = "LogicalOrOp"
    op_names = ["LogicalOr"]
    hw_config_names = [HWConfigOpName.LOGICALOR]


@OV_OPERATOR_METATYPES.register()
class OVLogicalXorMetatype(OVOpMetatype):
    name = "LogicalXorOp"
    op_names = ["LogicalXor"]
    hw_config_names = [HWConfigOpName.LOGICALXOR]


@OV_OPERATOR_METATYPES.register(is_subtype=True)
class OVEmbeddingMetatype(OVOpMetatype):
    name = "EmbeddingOp"
    hw_config_names = [HWConfigOpName.EMBEDDING]
    const_channel_axis = [0]

    @classmethod
    def matches(cls, node: ov.Node) -> bool:
        return _is_embedding(node)


@OV_OPERATOR_METATYPES.register()
class OVFloorMetatype(OVOpMetatype):
    name = "FloorOp"
    op_names = ["Floor"]


@OV_OPERATOR_METATYPES.register()
class OVFloorModMetatype(OVOpMetatype):
    name = "FloorModOp"
    op_names = ["FloorMod"]
    hw_config_names = [HWConfigOpName.FLOORMOD]


@OV_OPERATOR_METATYPES.register()
class OVMaximumMetatype(OVOpMetatype):
    name = "MaximumOp"
    op_names = ["Maximum"]
    hw_config_names = [HWConfigOpName.MAXIMUM]


@OV_OPERATOR_METATYPES.register()
class OVMinimumMetatype(OVOpMetatype):
    name = "MinimumOp"
    op_names = ["Minimum"]
    hw_config_names = [HWConfigOpName.MINIMUM]


@OV_OPERATOR_METATYPES.register()
class OVSqrtMetatype(OVOpMetatype):
    name = "SqrtOp"
    op_names = ["Sqrt"]
    hw_config_names = [HWConfigOpName.POWER]


@OV_OPERATOR_METATYPES.register()
class OVPowerMetatype(OVOpMetatype):
    name = "PowerOp"
    op_names = ["Power"]
    hw_config_names = [HWConfigOpName.POWER]


@OV_OPERATOR_METATYPES.register()
class OVLogMetatype(OVOpMetatype):
    name = "LogOp"
    op_names = ["Log"]


@OV_OPERATOR_METATYPES.register()
class OVROIAlignMetatype(OVOpMetatype):
    name = "ROIAlignOp"
    op_names = ["ROIAlign"]


@OV_OPERATOR_METATYPES.register()
class OVROIPoolingMetatype(OVOpMetatype):
    name = "ROIPoolingOp"
    op_names = ["ROIPooling"]


@OV_OPERATOR_METATYPES.register()
class OVGatherMetatype(OVOpMetatype):
    name = "GatherOp"
    op_names = ["Gather"]
    subtypes = [OVEmbeddingMetatype]


@OV_OPERATOR_METATYPES.register()
class OVGatherNDMetatype(OVOpMetatype):
    name = "GatherNDOp"
    op_names = ["GatherND"]


@OV_OPERATOR_METATYPES.register()
class OVGatherElementsMetatype(OVOpMetatype):
    name = "GatherElementsOp"
    op_names = ["GatherElements"]


@OV_OPERATOR_METATYPES.register()
class OVUnsqueezeMetatype(OVOpMetatype):
    name = "UnsqueezeOp"
    op_names = ["Unsqueeze"]
    hw_config_names = [HWConfigOpName.UNSQUEEZE]


@OV_OPERATOR_METATYPES.register()
class OVSqueezeMetatype(OVOpMetatype):
    name = "SqueezeOp"
    op_names = ["Squeeze"]
    hw_config_names = [HWConfigOpName.SQUEEZE]


@OV_OPERATOR_METATYPES.register()
class OVNonMaxSuppressionMetatype(OVOpMetatype):
    name = "NonMaxSuppressionOp"
    op_names = ["NonMaxSuppression"]


@OV_OPERATOR_METATYPES.register()
class OVReduceMinMetatype(OVOpMetatype):
    name = "ReduceMinOp"
    op_names = ["ReduceMin"]


@OV_OPERATOR_METATYPES.register()
class OVReduceMaxMetatype(OVOpMetatype):
    name = "ReduceMaxOp"
    op_names = ["ReduceMax"]
    hw_config_names = [HWConfigOpName.REDUCEMAX]


@OV_OPERATOR_METATYPES.register()
class OVReduceMeanMetatype(OVOpMetatype):
    name = "ReduceMeanOp"
    op_names = ["ReduceMean"]
    hw_config_names = [HWConfigOpName.REDUCEMEAN]


@OV_OPERATOR_METATYPES.register()
class OVReduceL1Metatype(OVOpMetatype):
    name = "ReduceL1Op"
    op_names = ["ReduceL1"]


@OV_OPERATOR_METATYPES.register()
class OVReduceL2Metatype(OVOpMetatype):
    name = "ReduceL2Op"
    op_names = ["ReduceL2"]
    hw_config_names = [HWConfigOpName.REDUCEL2]


@OV_OPERATOR_METATYPES.register()
class OVTopKMetatype(OVOpMetatype):
    name = "TopKOp"
    op_names = ["TopK"]


@OV_OPERATOR_METATYPES.register()
class OVStridedSliceMetatype(OVOpMetatype):
    name = "StridedSliceOp"
    op_names = ["StridedSlice"]
    hw_config_names = [HWConfigOpName.STRIDEDSLICE]


@OV_OPERATOR_METATYPES.register()
class OVSliceMetatype(OVOpMetatype):
    name = "SliceOp"
    op_names = ["Slice"]
    hw_config_names = [HWConfigOpName.SLICE]


@OV_OPERATOR_METATYPES.register()
class OVExpMetatype(OVOpMetatype):
    name = "ExpOp"
    op_names = ["Exp"]


@OV_OPERATOR_METATYPES.register()
class OVTransposeMetatype(OVOpMetatype):
    name = "TransposeOp"
    op_names = ["Transpose"]
    hw_config_names = [HWConfigOpName.TRANSPOSE]


@OV_OPERATOR_METATYPES.register()
class OVTileMetatype(OVOpMetatype):
    name = "TileOp"
    op_names = ["Tile"]
    hw_config_names = [HWConfigOpName.TILE]


@OV_OPERATOR_METATYPES.register()
class OVScatterElementsUpdateMetatype(OVOpMetatype):
    name = "ScatterElementsUpdateOp"
    op_names = ["ScatterElementsUpdate"]


@OV_OPERATOR_METATYPES.register()
class OVScatterNDUpdateMetatype(OVOpMetatype):
    name = "ScatterNDUpdateOp"
    op_names = ["ScatterNDUpdate"]


@OV_OPERATOR_METATYPES.register()
class OVScatterUpdateMetatype(OVOpMetatype):
    name = "ScatterUpdateOp"
    op_names = ["ScatterUpdate"]


@OV_OPERATOR_METATYPES.register()
class OVSoftmaxMetatype(OVOpMetatype):
    name = "SoftmaxOp"
    op_names = ["SoftMax", "Softmax"]


@OV_OPERATOR_METATYPES.register()
class OVPadMetatype(OVOpMetatype):
    name = "PadOp"
    op_names = ["Pad"]
    hw_config_names = [HWConfigOpName.PAD]


@OV_OPERATOR_METATYPES.register()
class OVReadValueMetatype(OVOpMetatype):
    name = "ReadValueOp"
    op_names = ["ReadValue"]


@OV_OPERATOR_METATYPES.register()
class OVAssignMetatype(OVOpMetatype):
    name = "AssignOp"
    op_names = ["Assign"]


@OV_OPERATOR_METATYPES.register()
class OVConvertMetatype(OVOpMetatype):
    name = "ConvertOp"
    op_names = ["Convert"]


@OV_OPERATOR_METATYPES.register()
@INPUT_NOOP_METATYPES.register()
class OVParameterMetatype(OVOpMetatype):
    name = "ParameterOp"
    op_names = ["Parameter"]


@OV_OPERATOR_METATYPES.register()
@OUTPUT_NOOP_METATYPES.register()
class OVResultMetatype(OVOpMetatype):
    name = "ResultOp"
    op_names = ["Result"]


@OV_OPERATOR_METATYPES.register()
class OVSwishMetatype(OVOpMetatype):
    name = "SwishOp"
    op_names = ["Swish"]


@OV_OPERATOR_METATYPES.register()
class OVHSwishMetatype(OVOpMetatype):
    name = "HSwishhOp"
    op_names = ["HSwish"]


@OV_OPERATOR_METATYPES.register()
class OVClampMetatype(OVOpMetatype):
    name = "ClampOp"
    op_names = ["Clamp"]


@OV_OPERATOR_METATYPES.register()
class OVSquaredDifferenceMetatype(OVOpMetatype):
    name = "SquaredDifferenceOp"
    op_names = ["SquaredDifference"]


@OV_OPERATOR_METATYPES.register()
class OVDeformableConvolutionMetatype(OVOpMetatype):
    name = "DeformableConvolutionOp"
    op_names = ["DeformableConvolution"]


@OV_OPERATOR_METATYPES.register()
class OVAbsMetatype(OVOpMetatype):
    name = "AbsOp"
    op_names = ["Abs"]


@OV_OPERATOR_METATYPES.register()
class OVIfMetatype(OVOpMetatype):
    name = "IfOp"
    op_names = ["If"]


@OV_OPERATOR_METATYPES.register()
class OVGroupNormalizationMetatype(OVOpMetatype):
    name = "GroupNormalizationOp"
    op_names = ["GroupNormalization"]
    hw_config_names = [HWConfigOpName.GROUPNORMALIZATION]


@OV_OPERATOR_METATYPES.register()
class OVScaledDotProductAttentionMetatype(OVOpMetatype):
    name = "ScaledDotProductAttentionOp"
    op_names = ["ScaledDotProductAttention"]
    hw_config_names = [HWConfigOpName.SCALED_DOT_PRODUCT_ATTENTION]
    target_input_ports = [0, 1]


@OV_OPERATOR_METATYPES.register()
class OVCosMetatype(OVOpMetatype):
    name = "CosOp"
    op_names = ["Cos"]


@OV_OPERATOR_METATYPES.register()
class OVSinMetatype(OVOpMetatype):
    name = "SinOp"
    op_names = ["Sin"]


def get_operator_metatypes() -> List[Type[OperatorMetatype]]:
    """
    Returns a list of the operator metatypes.
    :return: List of operator metatypes .
    """
    return list(OV_OPERATOR_METATYPES.registry_dict.values())


def get_operation_const_op(operation: ov.Node, const_port_id: int) -> Optional[ov.Node]:
    """
    Returns constant node of given operation placed on given const port id.

    :param operation: Given operation.
    :param const_port_id: Given constant port id.
    :returns: Constant node of given operation placed on given const port id.
    """
    node = operation.input_value(const_port_id).get_node()

    # There are several cases here
    # (Constant) -> (Operation)
    # (Constant) -> (Convert) -> (Operation)
    # (Constant) -> (Convert) -> (FakeQuantize, FakeConvert) -> (Operation)
    # (Constant) -> (Convert) -> (FakeQuantize, FakeConvert) -> (Reshape) -> (Operation)
    #  and etc. We need properly find the constant node. So we start with
    # `node` and traverse up until the constant node is not found.
    queue = deque([node])
    constant_node = None
    allowed_propagation_types_list = ["Convert", "FakeQuantize", "FakeConvert", "Reshape"]

    while len(queue) != 0:
        curr_node = queue.popleft()
        if curr_node.get_type_name() == "Constant":
            constant_node = curr_node
            break
        if len(curr_node.inputs()) == 0:
            break
        if curr_node.get_type_name() in allowed_propagation_types_list:
            queue.append(curr_node.input_value(0).get_node())

    return constant_node


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
    inp_channels = node.input_value(0).get_partial_shape().get_dimension(1)
    groups = node.input_value(1).get_partial_shape().get_dimension(0)
    if inp_channels.is_dynamic or groups.is_dynamic:
        return False
    inp_channels = inp_channels.get_length()
    groups = groups.get_length()
    return groups == inp_channels and inp_channels > 1


def _is_embedding(node: ov.Node) -> bool:
    """
    Returns True if the layer can be represented as embedding, False - otherwise.

    :param node: Layer to check whether it is embedding.
    :return: True if the layer is embedding, False - otherwise.
    """
    allowed_types_list = ["f16", "f32", "f64"]
    const_port_id = 0
    input_tensor = node.input_value(const_port_id)
    if input_tensor.get_element_type().get_type_name() in allowed_types_list:
        const_node = get_operation_const_op(node, const_port_id)
        if const_node is not None:
            return True

    return False


def get_node_metatype(node: ov.Node) -> Type[OperatorMetatype]:
    """
    Determine NNCF meta type for OpenVINO node.

    :param node: OpenVINO node.
    :return: NNCF meta type which corresponds to OpenVINO node.
    """
    node_type = node.get_type_name()
    metatype = OV_OPERATOR_METATYPES.get_operator_metatype_by_op_name(node_type)
    if metatype is not UnknownMetatype and metatype.get_subtypes():
        subtype = metatype.determine_subtype(node)
        if subtype is not None:
            metatype = subtype
    return metatype
