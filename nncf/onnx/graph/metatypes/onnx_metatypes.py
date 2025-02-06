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
from typing import Dict, List, Optional, Type

import onnx

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatypeRegistry
from nncf.common.hardware.opset import HWConfigOpName
from nncf.onnx.graph.onnx_helper import get_parent
from nncf.onnx.graph.onnx_helper import get_parents_node_mapping
from nncf.onnx.graph.onnx_helper import get_tensor
from nncf.onnx.graph.onnx_helper import has_tensor

ONNX_OPERATION_METATYPES = OperatorMetatypeRegistry("onnx_operator_metatypes")


class ONNXOpMetatype(OperatorMetatype):
    op_names: List[str] = []
    subtypes: List[Type[OperatorMetatype]] = []

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return cls.op_names

    @classmethod
    def get_subtypes(cls) -> List[Type[OperatorMetatype]]:
        return cls.subtypes

    @classmethod
    def matches(cls, model: onnx.ModelProto, node: onnx.NodeProto) -> bool:
        return node.op_type in cls.op_names

    @classmethod
    def determine_subtype(cls, model: onnx.ModelProto, node: onnx.NodeProto) -> Optional[Type[OperatorMetatype]]:
        matches = []
        subtypes_list = deque(cls.get_subtypes())
        while subtypes_list:
            subtype = subtypes_list.popleft()
            if subtype.matches(model, node):
                subtypes_list.extend(subtype.get_subtypes())
                matches.append(subtype)
        if not matches:
            return None
        return matches[-1]


class ONNXOpWithWeightsMetatype(ONNXOpMetatype):
    """
    Metatype which could have weights.
    :param weight_channel_axis: Axis for weight per-channel quantization.
    :param weight_port_ids: Constant input ports of the node's weight. Defaults to an empty list.
    :param bias_port_id: Input port of the node's bias. If the value is None,
    it means that the Metatype does not have bias. Defaults to None.
    :param possible_weight_ports: Input ports on which weight could be laid. Defaults to an empty list.
    """

    weight_channel_axis: int
    weight_port_ids: List[int] = []
    bias_port_id: Optional[int] = None
    possible_weight_ports: List[int] = []


@ONNX_OPERATION_METATYPES.register(is_subtype=True)
class ONNXDepthwiseConvolutionMetatype(ONNXOpWithWeightsMetatype):
    name = "DepthwiseConvOp"
    op_names = ["Conv"]
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]
    weight_channel_axis = 0
    weight_port_ids = [1]
    bias_port_id = 2
    output_channel_axis = 1

    @classmethod
    def matches(cls, model: onnx.ModelProto, node: onnx.NodeProto) -> bool:
        return _is_depthwise_conv(model, node)


@ONNX_OPERATION_METATYPES.register(is_subtype=True)
class ONNXGroupConvolutionMetatype(ONNXOpWithWeightsMetatype):
    name = "GroupConvOp"
    op_names = ["Conv"]
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    weight_channel_axis = 0
    weight_port_ids = [1]
    bias_port_id = 2
    output_channel_axis = 1
    subtypes = [ONNXDepthwiseConvolutionMetatype]

    @classmethod
    def matches(cls, model: onnx.ModelProto, node: onnx.NodeProto) -> bool:
        return _is_group_conv(node)


@ONNX_OPERATION_METATYPES.register()
class ONNXConvolutionMetatype(ONNXOpWithWeightsMetatype):
    name = "ConvOp"
    op_names = ["Conv"]
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    weight_channel_axis = 0
    weight_port_ids = [1]
    bias_port_id = 2
    output_channel_axis = 1
    subtypes = [ONNXGroupConvolutionMetatype]


@ONNX_OPERATION_METATYPES.register()
class ONNXConvolutionTransposeMetatype(ONNXOpWithWeightsMetatype):
    name = "ConvTransposeOp"
    op_names = ["ConvTranspose"]
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    weight_channel_axis = 1
    weight_port_ids = [1]
    bias_port_id = 2
    output_channel_axis = 1


@ONNX_OPERATION_METATYPES.register()
class ONNXGemmMetatype(ONNXOpWithWeightsMetatype):
    name = "GemmOp"
    op_names = ["Gemm"]
    hw_config_names = [HWConfigOpName.MATMUL]
    weight_channel_axis = -1  # For port_id=1
    bias_port_id = 2
    possible_weight_ports = [0, 1]
    output_channel_axis = -1


@ONNX_OPERATION_METATYPES.register()
class ONNXMatMulMetatype(ONNXOpWithWeightsMetatype):
    name = "MatMulOp"
    op_names = ["MatMul"]
    hw_config_names = [HWConfigOpName.MATMUL]
    weight_channel_axis = -1  # For port_id=1
    bias_port_id = 2
    possible_weight_ports = [0, 1]
    output_channel_axis = -1


@ONNX_OPERATION_METATYPES.register()
class ONNXReluMetatype(ONNXOpMetatype):
    name = "ReluOp"
    op_names = ["Relu", "Clip"]


@ONNX_OPERATION_METATYPES.register()
class ONNXLeakyReluMetatype(ONNXOpMetatype):
    name = "LeakyReluOp"
    op_names = ["LeakyRelu"]


@ONNX_OPERATION_METATYPES.register()
class ONNXThresholdedReluMetatype(ONNXOpMetatype):
    name = "ThresholdedReluOp"
    op_names = ["ThresholdedRelu"]


@ONNX_OPERATION_METATYPES.register()
class ONNXEluMetatype(ONNXOpMetatype):
    name = "EluOp"
    op_names = ["Elu"]


@ONNX_OPERATION_METATYPES.register()
class ONNXPReluMetatype(ONNXOpMetatype):
    name = "PReluOp"
    op_names = ["PRelu"]


@ONNX_OPERATION_METATYPES.register()
class ONNXSigmoidMetatype(ONNXOpMetatype):
    name = "SigmoidOp"
    op_names = ["Sigmoid"]


@ONNX_OPERATION_METATYPES.register()
class ONNXHardSigmoidMetatype(ONNXOpMetatype):
    name = "HardSigmoidOp"
    op_names = ["HardSigmoid"]


@ONNX_OPERATION_METATYPES.register()
class ONNXHardSwishMetatype(ONNXOpMetatype):
    name = "HardSwishOp"
    op_names = ["HardSwish"]


@ONNX_OPERATION_METATYPES.register()
class ONNXGlobalAveragePoolMetatype(ONNXOpMetatype):
    name = "GlobalAveragePoolOp"
    op_names = ["GlobalAveragePool"]
    hw_config_names = [HWConfigOpName.AVGPOOL]


@ONNX_OPERATION_METATYPES.register()
class ONNXAveragePoolMetatype(ONNXOpMetatype):
    name = "AveragePoolOp"
    op_names = ["AveragePool"]
    hw_config_names = [HWConfigOpName.AVGPOOL]


@ONNX_OPERATION_METATYPES.register()
class ONNXGlobalMaxPoolMetatype(ONNXOpMetatype):
    name = "GlobalMaxPoolOp"
    op_names = ["GlobalMaxPool"]
    hw_config_names = [HWConfigOpName.MAXPOOL]


@ONNX_OPERATION_METATYPES.register()
class ONNXMaxPoolMetatype(ONNXOpMetatype):
    name = "MaxPoolOp"
    op_names = ["MaxPool"]
    hw_config_names = [HWConfigOpName.MAXPOOL]


@ONNX_OPERATION_METATYPES.register()
class ONNXConstantMetatype(ONNXOpMetatype):
    name = "ConstantOp"
    op_names = ["Constant"]


@ONNX_OPERATION_METATYPES.register()
class ONNXAddLayerMetatype(ONNXOpMetatype):
    name = "AddOp"
    op_names = ["Add", "Sum"]
    hw_config_names = [HWConfigOpName.ADD]


@ONNX_OPERATION_METATYPES.register()
class ONNXSubMetatype(ONNXOpMetatype):
    name = "SubOp"
    op_names = ["Sub"]
    hw_config_names = [HWConfigOpName.SUBTRACT]


@ONNX_OPERATION_METATYPES.register()
class ONNXMulLayerMetatype(ONNXOpMetatype):
    name = "MulOp"
    op_names = ["Mul"]
    hw_config_names = [HWConfigOpName.MULTIPLY]


@ONNX_OPERATION_METATYPES.register()
class ONNXDivLayerMetatype(ONNXOpMetatype):
    name = "DivOp"
    op_names = ["Div"]
    hw_config_names = [HWConfigOpName.DIVIDE]


@ONNX_OPERATION_METATYPES.register()
class ONNXConcatMetatype(ONNXOpMetatype):
    name = "ConcatOp"
    op_names = ["Concat"]
    hw_config_names = [HWConfigOpName.CONCAT]


@ONNX_OPERATION_METATYPES.register()
class ONNXBatchNormMetatype(ONNXOpMetatype):
    name = "BatchNormalizationOp"
    op_names = ["BatchNormalization"]


@ONNX_OPERATION_METATYPES.register()
class ONNXResizeMetatype(ONNXOpMetatype):
    name = "ResizeOp"
    op_names = ["Resize"]
    hw_config_names = [HWConfigOpName.INTERPOLATE]


@ONNX_OPERATION_METATYPES.register()
class ONNXCenterCropPadMetatype(ONNXOpMetatype):
    name = "CenterCropPadOp"
    op_names = ["CenterCropPad"]
    hw_config_names = [HWConfigOpName.CROP]


@ONNX_OPERATION_METATYPES.register()
class ONNXReshapeMetatype(ONNXOpMetatype):
    name = "ReshapeOp"
    op_names = ["Reshape"]
    hw_config_names = [HWConfigOpName.RESHAPE]


@ONNX_OPERATION_METATYPES.register()
class ONNXTileMetatype(ONNXOpMetatype):
    name = "TileOp"
    op_names = ["Tile"]


@ONNX_OPERATION_METATYPES.register()
class ONNXUpsampleMetatype(ONNXOpMetatype):
    name = "UpsampleOp"
    op_names = ["Upsample"]


@ONNX_OPERATION_METATYPES.register()
class ONNXConstantOfShapeMetatype(ONNXOpMetatype):
    name = "ConstantOfShapeOp"
    op_names = ["ConstantOfShape"]


@ONNX_OPERATION_METATYPES.register()
class ONNXShapeMetatype(ONNXOpMetatype):
    name = "ShapeOp"
    op_names = ["Shape"]


@ONNX_OPERATION_METATYPES.register()
class ONNXExpandMetatype(ONNXOpMetatype):
    name = "ExpandOp"
    op_names = ["Expand"]


@ONNX_OPERATION_METATYPES.register()
class ONNXNonZeroMetatype(ONNXOpMetatype):
    name = "NonZeroOp"
    op_names = ["NonZero"]


@ONNX_OPERATION_METATYPES.register()
class ONNXSplitMetatype(ONNXOpMetatype):
    name = "SplitOp"
    op_names = ["Split"]
    hw_config_names = [HWConfigOpName.SPLIT]


@ONNX_OPERATION_METATYPES.register()
class ONNXLessMetatype(ONNXOpMetatype):
    name = "LessOp"
    op_names = ["Less"]
    hw_config_names = [HWConfigOpName.LESS]


@ONNX_OPERATION_METATYPES.register()
class ONNXLessOrEqualMetatype(ONNXOpMetatype):
    name = "LessOrEqualOp"
    op_names = ["LessOrEqual"]
    hw_config_names = [HWConfigOpName.LESSEQUAL]


@ONNX_OPERATION_METATYPES.register()
class ONNXGreaterMetatype(ONNXOpMetatype):
    name = "GreaterOp"
    op_names = ["Greater"]
    hw_config_names = [HWConfigOpName.GREATER]


@ONNX_OPERATION_METATYPES.register()
class ONNXGreaterOrEqualMetatype(ONNXOpMetatype):
    name = "GreaterOrEqualOp"
    op_names = ["GreaterOrEqual"]
    hw_config_names = [HWConfigOpName.GREATEREQUAL]


@ONNX_OPERATION_METATYPES.register()
class ONNXEqualMetatype(ONNXOpMetatype):
    name = "EqualOp"
    op_names = ["Equal"]
    hw_config_names = [HWConfigOpName.EQUAL]


@ONNX_OPERATION_METATYPES.register()
class ONNXNotMetatype(ONNXOpMetatype):
    name = "NotOp"
    op_names = ["Not"]
    hw_config_names = [HWConfigOpName.LOGICALNOT]


@ONNX_OPERATION_METATYPES.register()
class ONNXAndMetatype(ONNXOpMetatype):
    name = "AndOp"
    op_names = ["And"]
    hw_config_names = [HWConfigOpName.LOGICALAND]


@ONNX_OPERATION_METATYPES.register()
class ONNXOrMetatype(ONNXOpMetatype):
    name = "OrOp"
    op_names = ["Or"]
    hw_config_names = [HWConfigOpName.LOGICALOR]


@ONNX_OPERATION_METATYPES.register()
class ONNXXOrMetatype(ONNXOpMetatype):
    name = "XorOp"
    op_names = ["Xor"]
    hw_config_names = [HWConfigOpName.LOGICALXOR]


@ONNX_OPERATION_METATYPES.register()
class ONNXModMetatype(ONNXOpMetatype):
    name = "ModOp"
    op_names = ["Mod"]
    hw_config_names = [HWConfigOpName.FLOORMOD]


@ONNX_OPERATION_METATYPES.register()
class ONNXMaximumMetatype(ONNXOpMetatype):
    name = "MaxOp"
    op_names = ["Max"]
    hw_config_names = [HWConfigOpName.MAXIMUM]


@ONNX_OPERATION_METATYPES.register()
class ONNXMinimumMetatype(ONNXOpMetatype):
    name = "MinOp"
    op_names = ["Min"]
    hw_config_names = [HWConfigOpName.MINIMUM]


@ONNX_OPERATION_METATYPES.register()
class ONNXMeanMetatype(ONNXOpMetatype):
    name = "MeanOp"
    op_names = ["Mean"]


@ONNX_OPERATION_METATYPES.register()
class ONNXFloorMetatype(ONNXOpMetatype):
    name = "FloorOp"
    op_names = ["Floor"]


@ONNX_OPERATION_METATYPES.register()
class ONNXPowMetatype(ONNXOpMetatype):
    name = "PowOp"
    op_names = ["Pow"]
    hw_config_names = [HWConfigOpName.POWER]


@ONNX_OPERATION_METATYPES.register()
class ONNXSqrtMetatype(ONNXOpMetatype):
    name = "SqrtOp"
    op_names = ["Sqrt"]
    hw_config_names = [HWConfigOpName.POWER]


@ONNX_OPERATION_METATYPES.register()
class ONNXReciprocalMetatype(ONNXOpMetatype):
    name = "ReciprocalOp"
    op_names = ["Reciprocal"]
    hw_config_names = [HWConfigOpName.POWER]


@ONNX_OPERATION_METATYPES.register(is_subtype=True)
class ONNXEmbeddingMetatype(ONNXOpWithWeightsMetatype):
    name = "EmbeddingOp"
    hw_config_names = [HWConfigOpName.EMBEDDING]
    weight_port_ids = [0]
    weight_channel_axis = 0

    @classmethod
    def matches(cls, model: onnx.ModelProto, node: onnx.NodeProto) -> bool:
        return _is_embedding(model, node)


@ONNX_OPERATION_METATYPES.register()
class ONNXLogMetatype(ONNXOpMetatype):
    name = "LogOp"
    op_names = ["Log"]


@ONNX_OPERATION_METATYPES.register()
class ONNXAbsMetatype(ONNXOpMetatype):
    name = "AbsOp"
    op_names = ["Abs"]


@ONNX_OPERATION_METATYPES.register()
class ONNXScatterElementsMetatype(ONNXOpMetatype):
    name = "ScatterElementsOp"
    op_names = ["ScatterElements"]


@ONNX_OPERATION_METATYPES.register()
class ONNXScatterMetatype(ONNXOpMetatype):
    name = "ScatterOp"
    op_names = ["Scatter"]


@ONNX_OPERATION_METATYPES.register()
class ONNXScatterNDMetatype(ONNXOpMetatype):
    name = "ScatterNDOp"
    op_names = ["ScatterND"]


@ONNX_OPERATION_METATYPES.register()
class ONNXROIAlignMetatype(ONNXOpMetatype):
    name = "ROIAlignOp"
    op_names = ["RoiAlign"]


@ONNX_OPERATION_METATYPES.register()
class ONNXGatherMetatype(ONNXOpMetatype):
    name = "GatherOp"
    op_names = ["Gather"]
    subtypes = [ONNXEmbeddingMetatype]


@ONNX_OPERATION_METATYPES.register()
class ONNXGatherNDMetatype(ONNXOpMetatype):
    name = "GatherNDOp"
    op_names = ["GatherND"]


@ONNX_OPERATION_METATYPES.register()
class ONNXGatherElementsMetatype(ONNXOpMetatype):
    name = "GatherElementsOp"
    op_names = ["GatherElements"]


@ONNX_OPERATION_METATYPES.register()
class ONNXUnsqueezeMetatype(ONNXOpMetatype):
    name = "UnsqueezeOp"
    op_names = ["Unsqueeze"]
    hw_config_names = [HWConfigOpName.UNSQUEEZE]


@ONNX_OPERATION_METATYPES.register()
class ONNXSqueezeMetatype(ONNXOpMetatype):
    name = "SqueezeOp"
    op_names = ["Squeeze"]
    hw_config_names = [HWConfigOpName.SQUEEZE]


@ONNX_OPERATION_METATYPES.register()
class ONNXNonMaxSuppressionMetatype(ONNXOpMetatype):
    name = "NonMaxSuppressionOp"
    op_names = ["NonMaxSuppression"]


@ONNX_OPERATION_METATYPES.register()
class ONNXCastMetatype(ONNXOpMetatype):
    name = "CastOp"
    op_names = ["Cast"]


@ONNX_OPERATION_METATYPES.register()
class ONNXCastLikeMetatype(ONNXOpMetatype):
    name = "CastLikeOp"
    op_names = ["CastLike"]


@ONNX_OPERATION_METATYPES.register()
class ONNXReduceMinMetatype(ONNXOpMetatype):
    name = "ReduceMinOp"
    op_names = ["ReduceMin"]


@ONNX_OPERATION_METATYPES.register()
class ONNXReduceMaxMetatype(ONNXOpMetatype):
    name = "ReduceMaxOp"
    op_names = ["ReduceMax"]
    hw_config_names = [HWConfigOpName.REDUCEMAX]


@ONNX_OPERATION_METATYPES.register()
class ONNXReduceSumMetatype(ONNXOpMetatype):
    name = "ReduceSumOp"
    op_names = ["ReduceSum"]
    hw_config_names = [HWConfigOpName.REDUCESUM]


class ONNXReduceL2Metatype(ONNXOpMetatype):
    name = "ReduceL2Op"
    op_names = ["ReduceL2"]
    hw_config_names = [HWConfigOpName.REDUCEL2]


@ONNX_OPERATION_METATYPES.register()
class ONNXDepthToSpaceMetatype(ONNXOpMetatype):
    name = "DepthToSpaceOp"
    op_names = ["DepthToSpace"]


@ONNX_OPERATION_METATYPES.register()
class ONNXSpaceToDepthMetatype(ONNXOpMetatype):
    name = "SpaceToDepthOp"
    op_names = ["SpaceToDepth"]


@ONNX_OPERATION_METATYPES.register()
class ONNXReduceMeanMetatype(ONNXOpMetatype):
    name = "ReduceMeanOp"
    op_names = ["ReduceMean"]
    hw_config_names = [HWConfigOpName.REDUCEMEAN]


@ONNX_OPERATION_METATYPES.register()
class ONNXTopKMetatype(ONNXOpMetatype):
    name = "TopKOp"
    op_names = ["TopK"]


@ONNX_OPERATION_METATYPES.register()
class ONNXSliceMetatype(ONNXOpMetatype):
    name = "SliceOp"
    op_names = ["Slice"]


@ONNX_OPERATION_METATYPES.register()
class ONNXExpMetatype(ONNXOpMetatype):
    name = "ExpOp"
    op_names = ["Exp"]


@ONNX_OPERATION_METATYPES.register()
class ONNXTransposeMetatype(ONNXOpMetatype):
    name = "TransposeOp"
    op_names = ["Transpose"]
    hw_config_names = [HWConfigOpName.TRANSPOSE]


@ONNX_OPERATION_METATYPES.register()
class ONNXDropoutMetatype(ONNXOpMetatype):
    name = "DropoutOp"
    op_names = ["Dropout"]


@ONNX_OPERATION_METATYPES.register()
class ONNXFlattenMetatype(ONNXOpMetatype):
    name = "FlattenOp"
    op_names = ["Flatten"]
    hw_config_names = [HWConfigOpName.FLATTEN]


@ONNX_OPERATION_METATYPES.register()
class ONNXSoftmaxMetatype(ONNXOpMetatype):
    name = "SoftmaxOp"
    op_names = ["Softmax"]


@ONNX_OPERATION_METATYPES.register()
class ONNXPadMetatype(ONNXOpMetatype):
    name = "PadOp"
    op_names = ["Pad"]


@ONNX_OPERATION_METATYPES.register()
class ONNXIdentityMetatype(ONNXOpMetatype):
    name = "IdentityOp"
    op_names = ["Identity"]


@ONNX_OPERATION_METATYPES.register()
class ONNXQuantizeLinearMetatype(ONNXOpMetatype):
    name = "QuantizeLinearOp"
    op_names = ["QuantizeLinear"]


@ONNX_OPERATION_METATYPES.register()
class ONNXDequantizeLinearMetatype(ONNXOpMetatype):
    name = "DequantizeLinearOp"
    op_names = ["DequantizeLinear"]


@ONNX_OPERATION_METATYPES.register()
class ONNXDeformableConvolutionMetatype(ONNXOpMetatype):
    name = "DeformConvOp"
    op_names = ["DeformConv"]


@ONNX_OPERATION_METATYPES.register()
class ONNXErfMetatype(ONNXOpMetatype):
    name = "ErfOp"
    op_names = ["Erf"]


@ONNX_OPERATION_METATYPES.register()
class ONNXCosMetatype(ONNXOpMetatype):
    name = "CosOp"
    op_names = ["Cos"]


@ONNX_OPERATION_METATYPES.register()
class ONNXSinMetatype(ONNXOpMetatype):
    name = "SinOp"
    op_names = ["Sin"]


def get_operator_metatypes() -> List[Type[OperatorMetatype]]:
    """
    Returns a list of the operator metatypes.

    :return: List of operator metatypes.
    """
    return list(ONNX_OPERATION_METATYPES.registry_dict.values())


def get_metatype(model: onnx.ModelProto, node: onnx.NodeProto) -> ONNXOpMetatype:
    """
    Returns matched ONNXOpMetatype metatype to a ONNX node.

    :param model: ONNX model.
    :param node: Node from ONNX model.
    :return: Matched metatype.
    """
    metatype = ONNX_OPERATION_METATYPES.get_operator_metatype_by_op_name(node.op_type)
    if metatype.get_subtypes():
        subtype = metatype.determine_subtype(model, node)
        if subtype is not None:
            metatype = subtype
    return metatype


def get_tensor_edge_name(
    model: onnx.ModelProto,
    node: onnx.NodeProto,
    port_id: int,
    parents_node_mapping: Dict[str, onnx.NodeProto],
) -> Optional[str]:
    """
    Returns an edge name associated with a weight of a node laying on  an input port_id.

    Checks whether a node has a tensor on input port_id.
    If does then it is a weight and returns corresponding edge name.
    If not - take a parent node into this port id and does the same check for it.

    If an edge with a weight was not found then returns None.

    METATYPES THAT COULD CONSUME A WEIGHT TENSOR:
        ONNXConstantMetatype
        ONNXIdentityMetatype
        ONNXReshapeMetatype
        ONNXTransposeMetatype
        ONNXQuantizeLinearMetatype

    :param model: ONNX model.
    :param node: Node.
    :param port_id: Port id on which a weight edge is seeking.
    :param parents_node_mapping: Mapping from edge name to node which outputs this edge.
    :return: Edge name associated with a weight.
    """
    PROPAGATING_NODES = (
        ONNXIdentityMetatype.get_all_aliases()
        + ONNXTransposeMetatype.get_all_aliases()
        + ONNXQuantizeLinearMetatype.get_all_aliases()
        + ONNXReshapeMetatype.get_all_aliases()
        + ONNXDequantizeLinearMetatype.get_all_aliases()
    )
    END_NODES = ONNXConstantMetatype.get_all_aliases()
    parent = get_parent(node, port_id, parents_node_mapping)
    if not parent:
        if has_tensor(model, node.input[port_id]):
            return node.input[port_id]
    elif parent.op_type in END_NODES:
        return node.input[port_id]
    elif parent.op_type in PROPAGATING_NODES:
        return get_tensor_edge_name(model, parent, 0, parents_node_mapping)
    return None


def _is_group_conv(node: onnx.NodeProto) -> bool:
    """
    Returns True if the convolution is group, False - otherwise.
    Group convolution is a convolution with the group attribute.

    :param node: Convolution node to check whether it is depthwise.
    :return: True if the convolution is group, False - otherwise.
    """
    conv_group = None
    for attribute in node.attribute:
        if attribute.name == "group":
            conv_group = onnx.helper.get_attribute_value(attribute)
    if conv_group is None or conv_group == 1:
        return False
    return True


def _is_depthwise_conv(model: onnx.ModelProto, node: onnx.NodeProto) -> bool:
    """
    Returns True if the convolution is depthwise, False - otherwise.
    Depthwise convolution is a convolution satisfies the following rule:
    groups == in_channels and out_channels == K*in_channels, where K is a positive integer.
    Weight tensor of a convolution consists of the following dimension:
    (out_channels, in_channels / groups, kernel_size[0], kernel_size[1]).

    :param model: ONNX model to get the node's weight.
    :param node: Convolution node to check whether it is depthwise.
    :return: True if the convolution is depthwise, False - otherwise.
    """
    for attribute in node.attribute:
        if attribute.name == "group":
            conv_group = onnx.helper.get_attribute_value(attribute)
    weight_tensor_value = None
    initializer_name = get_tensor_edge_name(model, node, 1, get_parents_node_mapping(model))
    for init in model.graph.initializer:
        if init.name == initializer_name:
            weight_tensor_value = onnx.numpy_helper.to_array(init)
    if weight_tensor_value is None:
        return False
    conv_out_channels = weight_tensor_value.shape[0]
    conv_in_channels = weight_tensor_value.shape[1] * conv_group
    if (
        conv_out_channels % conv_in_channels == 0
        and conv_out_channels // conv_in_channels > 0
        and conv_group == conv_in_channels
    ):
        return True
    return False


def _is_embedding(model: onnx.ModelProto, node: onnx.NodeProto) -> bool:
    """
    Returns True if the layer can be represented as embedding, False - otherwise.

    :param model: ONNX model to get the node's weight.
    :param node: Layer to check whether it is embedding.
    :return: True if the layer is embedding, False - otherwise.
    """
    tensor_port_id = ONNXEmbeddingMetatype.weight_port_ids[0]
    allowed_types_list = ["TensorProto.FLOAT"]
    parents_node_mapping = get_parents_node_mapping(model)
    weight_edge_name = get_tensor_edge_name(model, node, tensor_port_id, parents_node_mapping)

    if weight_edge_name is not None:
        tensor_data_type = get_tensor(model, weight_edge_name).data_type
        if onnx.helper.tensor_dtype_to_string(tensor_data_type) in allowed_types_list:
            return True
    return False
