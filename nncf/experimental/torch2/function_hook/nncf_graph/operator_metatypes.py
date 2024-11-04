# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, TypeVar, cast

import nncf
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.operator_metatypes import CONST_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import INPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OUTPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatypeRegistry
from nncf.common.hardware.opset import HWConfigOpName
from nncf.experimental.torch2.function_hook.graph.graph_utils import TensorMeta

ModuleAttributes = TypeVar("ModuleAttributes", bound=BaseLayerAttributes)

PT_OPERATOR_METATYPES = OperatorMetatypeRegistry("operator_metatypes")


class PTOperatorMetatype(OperatorMetatype):
    """
    Base class for grouping PyTorch operators based on their semantic meaning.
    Grouping also allows efficient application of HW specifics to compression of
    certain operation groups.
    """

    op_names: List[str] = []
    subtypes: list[type[OperatorMetatype]] = []

    num_expected_input_edges: Optional[int] = None
    weight_port_ids: List[int] = []

    @classmethod
    def get_subtypes(cls) -> List[type[OperatorMetatype]]:
        return cls.subtypes.copy()

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return cls.op_names.copy()

    @classmethod
    def determine_subtype(cls, fn_args: List[Any], fn_kwargs: Dict[str, Any]) -> Optional[type[OperatorMetatype]]:
        """
        Determines the subtype of the operator based on the given function arguments and keyword arguments.

        :param fn_args: The function arguments.
        :param fn_kwargs: The function keyword arguments.
        :return: The determined subtype of the operator, or None if no subtype is found.
        """
        matches: List[type[OperatorMetatype]] = []
        for subtype in cls.subtypes:
            if subtype.matches(fn_args, fn_kwargs):  # type: ignore[attr-defined]
                matches.append(subtype)
        if len(matches) > 1:
            raise nncf.InternalError("Multiple subtypes match operator call - cannot determine single subtype.")
        if not matches:
            return None
        return matches[0]


class PTOperatorSubtype(PTOperatorMetatype):
    """
    Exact specialization of PTOperatorMetatype that can only be determined via operator argument
    inspection or owning module attribute inspection, and that may have specialized compression method
    configuration other than the one used for general operations having the type of PTOperatorMetatype.
    """

    @classmethod
    def matches(cls, fn_args: List[Any], fn_kwargs: Dict[str, Any]) -> bool:
        """
        Check if the given function arguments and keyword arguments match the expected pattern.

        :param fn_args: A list of function arguments.
        :param fn_kwargs: A dictionary of function keyword arguments.
        :return: True if the arguments match the expected pattern, False otherwise.
        """
        raise NotImplementedError


@PT_OPERATOR_METATYPES.register()
@INPUT_NOOP_METATYPES.register()
class PTInputNoopMetatype(PTOperatorMetatype):
    name = NNCFGraphNodeType.OUTPUT_NODE
    op_names = [NNCFGraphNodeType.INPUT_NODE]


@PT_OPERATOR_METATYPES.register()
@OUTPUT_NOOP_METATYPES.register()
class PTOutputNoopMetatype(PTOperatorMetatype):
    name = NNCFGraphNodeType.OUTPUT_NODE
    op_names = [NNCFGraphNodeType.OUTPUT_NODE]


@PT_OPERATOR_METATYPES.register()
@CONST_NOOP_METATYPES.register()
class PTConstNoopMetatype(PTOperatorMetatype):
    name = NNCFGraphNodeType.OUTPUT_NODE
    op_names = [NNCFGraphNodeType.CONST_NODE]


@PT_OPERATOR_METATYPES.register()
@NOOP_METATYPES.register()
class PTNoopMetatype(PTOperatorMetatype):
    name = "noop"
    op_names = ["contiguous", "clone", "detach", "detach_", "to"]


class PTDepthwiseConvOperatorSubtype(PTOperatorSubtype):
    @classmethod
    def matches(cls, fn_args: List[Any], fn_kwargs: Dict[str, Any]) -> bool:
        weight_meta = cast(TensorMeta, fn_kwargs.get("weight", fn_args[0]))
        in_channels = weight_meta.shape[1]
        groups = fn_kwargs.get("groups", fn_args[7] if len(fn_args) > 7 else 1)
        return in_channels > 1 and groups == in_channels


@PT_OPERATOR_METATYPES.register(is_subtype=True)
class PTDepthwiseConv1dSubtype(PTDepthwiseConvOperatorSubtype):
    name = "Conv1DOp"
    hw_config_name = [HWConfigOpName.DEPTHWISECONVOLUTION]
    op_names = ["conv1d"]
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@PT_OPERATOR_METATYPES.register()
class PTConv1dMetatype(PTOperatorMetatype):
    name = "Conv1DOp"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    op_names = ["conv1d"]
    subtypes = [PTDepthwiseConv1dSubtype]
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@PT_OPERATOR_METATYPES.register(is_subtype=True)
class PTDepthwiseConv2dSubtype(PTDepthwiseConvOperatorSubtype):
    name = "Conv2DOp"
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]
    op_names = ["conv2d"]
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@PT_OPERATOR_METATYPES.register()
class PTConv2dMetatype(PTOperatorMetatype):
    name = "Conv2DOp"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    op_names = ["conv2d"]
    subtypes = [PTDepthwiseConv2dSubtype]
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@PT_OPERATOR_METATYPES.register(is_subtype=True)
class PTDepthwiseConv3dSubtype(PTDepthwiseConvOperatorSubtype):
    name = "Conv3DOp"
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]
    op_names = ["conv3d"]
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@PT_OPERATOR_METATYPES.register()
class PTConv3dMetatype(PTOperatorMetatype):
    name = "Conv3DOp"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    op_names = ["conv3d"]
    subtypes = [PTDepthwiseConv3dSubtype]
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@PT_OPERATOR_METATYPES.register()
class PTConvTranspose1dMetatype(PTOperatorMetatype):
    name = "ConvTranspose1DOp"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    op_names = ["conv_transpose1d"]
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@PT_OPERATOR_METATYPES.register()
class PTConvTranspose2dMetatype(PTOperatorMetatype):
    name = "ConvTranspose2DOp"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    op_names = ["conv_transpose2d"]
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@PT_OPERATOR_METATYPES.register()
class PTConvTranspose3dMetatype(PTOperatorMetatype):
    name = "ConvTranspose3DOp"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    op_names = ["conv_transpose3d"]
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@PT_OPERATOR_METATYPES.register()
class PTDeformConv2dMetatype(PTOperatorMetatype):
    name = "DeformConv2dOp"
    op_names = ["deform_conv2d"]
    num_expected_input_edges = 4
    weight_port_ids = [2]


@PT_OPERATOR_METATYPES.register()
class PTLinearMetatype(PTOperatorMetatype):
    name = "LinearOp"
    op_names = ["linear"]
    hw_config_names = [HWConfigOpName.MATMUL]
    output_channel_axis = -1
    num_expected_input_edges = 2
    weight_port_ids = [1]


@PT_OPERATOR_METATYPES.register()
class PTHardTanhMetatype(PTOperatorMetatype):
    name = "HardTanhOP"
    op_names = ["hardtanh"]
    num_expected_input_edges = 1


@PT_OPERATOR_METATYPES.register()
class PTHardSwishMetatype(PTOperatorMetatype):
    name = "HardSwishOp"
    op_names = ["hardswish", "hardswish_"]
    num_expected_input_edges = 1


@PT_OPERATOR_METATYPES.register()
class PTHardSigmoidMetatype(PTOperatorMetatype):
    name = "HardSigmoidOp"
    op_names = ["hardsigmoid"]
    num_expected_input_edges = 1


@PT_OPERATOR_METATYPES.register()
class PTTanhMetatype(PTOperatorMetatype):
    name = "TanhOp"
    op_names = ["tanh"]
    num_expected_input_edges = 1


@PT_OPERATOR_METATYPES.register()
class PTELUMetatype(PTOperatorMetatype):
    name = "EluOp"
    op_names = ["elu", "elu_"]
    num_expected_input_edges = 1


@PT_OPERATOR_METATYPES.register()
class PTPRELUMetatype(PTOperatorMetatype):
    name = "PReluOp"
    op_names = ["prelu"]
    num_expected_input_edges = 1


@PT_OPERATOR_METATYPES.register()
class PTLeakyRELUMetatype(PTOperatorMetatype):
    name = "LeakyReluOp"
    op_names = ["leaky_relu"]
    num_expected_input_edges = 1


@PT_OPERATOR_METATYPES.register()
class PTLayerNormMetatype(PTOperatorMetatype):
    name = "LayerNormOp"
    op_names = ["layer_norm"]
    hw_config_names = [HWConfigOpName.MVN]
    num_expected_input_edges = 1
    weight_port_ids = [2]


@PT_OPERATOR_METATYPES.register()
class PTGroupNormMetatype(PTOperatorMetatype):
    name = "GroupNormOp"
    op_names = ["group_norm"]
    hw_config_names = [HWConfigOpName.MVN]
    weight_port_ids = [2]


@PT_OPERATOR_METATYPES.register()
class PTGELUMetatype(PTOperatorMetatype):
    name = "GeluOp"
    op_names = ["gelu"]
    hw_config_names = [HWConfigOpName.GELU]


@PT_OPERATOR_METATYPES.register()
class PTSILUMetatype(PTOperatorMetatype):
    name = "SiluOp"
    op_names = ["silu", "silu_"]


@PT_OPERATOR_METATYPES.register()
class PTSigmoidMetatype(PTOperatorMetatype):
    name = "SigmoidOp"
    op_names = ["sigmoid"]


@PT_OPERATOR_METATYPES.register()
class PTAddMetatype(PTOperatorMetatype):
    name = "AddOp"
    op_names = ["add", "add_", "__add__", "__iadd__", "__radd__"]
    hw_config_names = [HWConfigOpName.ADD]
    num_expected_input_edges = 2


@PT_OPERATOR_METATYPES.register()
class PTSubMetatype(PTOperatorMetatype):
    name = "SubOp"
    op_names = ["sub", "sub_", "__sub__", "__isub__", "__rsub__"]
    hw_config_names = [HWConfigOpName.SUBTRACT]
    num_expected_input_edges = 2


@PT_OPERATOR_METATYPES.register()
class PTMulMetatype(PTOperatorMetatype):
    name = "MulOp"
    op_names = ["mul", "mul_", "__mul__", "__imul__", "__rmul__"]
    hw_config_names = [HWConfigOpName.MULTIPLY]
    num_expected_input_edges = 2


@PT_OPERATOR_METATYPES.register()
class PTDivMetatype(PTOperatorMetatype):
    name = "DivOp"
    op_names = ["div", "div_", "__div__", "__idiv__", "__rdiv__", "__truediv__", "__itruediv__", "__rtruediv__"]
    hw_config_names = [HWConfigOpName.DIVIDE]
    num_expected_input_edges = 2


@PT_OPERATOR_METATYPES.register()
class PTFloorDivMetatype(PTOperatorMetatype):
    name = "FloordivOp"
    op_names = ["floor_divide", "__floordiv__", "__ifloordiv__", "__rfloordiv__"]
    num_expected_input_edges = 2


@PT_OPERATOR_METATYPES.register()
class PTExpMetatype(PTOperatorMetatype):
    name = "ExpOp"
    op_names = ["exp", "exp_"]


@PT_OPERATOR_METATYPES.register()
class PTLogMetatype(PTOperatorMetatype):
    name = "LogOp"
    op_names = ["log", "log_"]


@PT_OPERATOR_METATYPES.register()
class PTAbsMetatype(PTOperatorMetatype):
    name = "AbsOp"
    op_names = ["abs", "abs_", "__abs__"]


@PT_OPERATOR_METATYPES.register()
class PTErfMetatype(PTOperatorMetatype):
    name = "ErfOp"
    op_names = ["erf", "erf_"]


@PT_OPERATOR_METATYPES.register()
class PTMatMulMetatype(PTOperatorMetatype):
    name = "MatMulOp"
    op_names = ["matmul", "bmm", "mm", "__matmul__", "__rmatmul__"]
    hw_config_names = [HWConfigOpName.MATMUL]
    num_expected_input_edges = 2
    weight_port_ids = [0, 1]


@PT_OPERATOR_METATYPES.register()
class PTAddmmMetatype(PTOperatorMetatype):
    name = "MatMulOp"
    op_names = ["addmm", "baddbmm"]
    hw_config_names = [HWConfigOpName.MATMUL]
    # 0-th arg to the baddbmm is basically a (b)ias to be (add)ed to the (bmm) operation,
    # presuming that most runtime implementations will fuse the bias addition into the matrix multiplication
    # and therefore won't quantize the bias input, as this would break the hardware-fused pattern.
    ignored_input_ports: List[int] = [0]
    num_expected_input_edges = 2
    weight_port_ids = [1, 2]


@PT_OPERATOR_METATYPES.register()
class PTMeanMetatype(PTOperatorMetatype):
    name = "MeanOp"
    op_names = ["mean"]
    hw_config_names = [HWConfigOpName.REDUCEMEAN]


@PT_OPERATOR_METATYPES.register()
class PTRoundMetatype(PTOperatorMetatype):
    name = "RoundOp"
    op_names = ["round", "round_"]


@PT_OPERATOR_METATYPES.register()
class PTDropoutMetatype(PTOperatorMetatype):
    name = "DropoutOp"
    op_names = ["dropout", "dropout_"]


@PT_OPERATOR_METATYPES.register()
class PTThresholdMetatype(PTOperatorMetatype):
    name = "ThresholdOp"
    op_name = ["threshold"]


@PT_OPERATOR_METATYPES.register()
class PTBatchNormMetatype(PTOperatorMetatype):
    name = "BatchNormOp"
    op_names = ["batch_norm", "batch_norm_"]
    weight_port_ids = [3]
    bias_port_id = 4


@PT_OPERATOR_METATYPES.register()
class PTAvgPool2dMetatype(PTOperatorMetatype):
    name = "AvgPool2DOp"
    op_names = ["avg_pool2d", "adaptive_avg_pool2d"]
    hw_config_names = [HWConfigOpName.AVGPOOL]


@PT_OPERATOR_METATYPES.register()
class PTAvgPool3dMetatype(PTOperatorMetatype):
    name = "AvgPool3DOp"
    op_names = ["avg_pool3d", "adaptive_avg_pool3d"]
    hw_config_names = [HWConfigOpName.AVGPOOL]


class PTAdaptiveMaxPool1dMetatype(PTOperatorMetatype):
    name = "AdaptiveMaxPool1DOp"
    op_names = ["adaptive_max_pool1d"]


@PT_OPERATOR_METATYPES.register()
class PTAdaptiveMaxPool2dMetatype(PTOperatorMetatype):
    name = "AdaptiveMaxPool2DOp"
    op_names = ["adaptive_max_pool2d"]


@PT_OPERATOR_METATYPES.register()
class PTAdaptiveMaxPool3dMetatype(PTOperatorMetatype):
    name = "AdaptiveMaxPool3DOp"
    op_names = ["adaptive_max_pool3d"]


class PTMaxPool1dMetatype(PTOperatorMetatype):
    name = "MaxPool1DOp"
    op_names = ["max_pool1d"]
    hw_config_names = [HWConfigOpName.MAXPOOL]


@PT_OPERATOR_METATYPES.register()
class PTMaxPool2dMetatype(PTOperatorMetatype):
    name = "MaxPool2DOp"
    op_names = ["max_pool2d"]
    hw_config_names = [HWConfigOpName.MAXPOOL]


@PT_OPERATOR_METATYPES.register()
class PTMaxPool3dMetatype(PTOperatorMetatype):
    name = "MaxPool3DOp"
    op_names = ["max_pool3d"]
    hw_config_names = [HWConfigOpName.MAXPOOL]


@PT_OPERATOR_METATYPES.register()
class PTMaxUnpool1dMetatype(PTOperatorMetatype):
    name = "MaxUnPool1DOp"
    op_names = ["max_unpool1d"]


@PT_OPERATOR_METATYPES.register()
class PTMaxUnpool2dMetatype(PTOperatorMetatype):
    name = "MaxUnPool2DOp"
    op_names = ["max_unpool2d"]


@PT_OPERATOR_METATYPES.register()
class PTMaxUnpool3dMetatype(PTOperatorMetatype):
    name = "MaxUnPool3DOp"
    op_names = ["max_unpool3d"]


@PT_OPERATOR_METATYPES.register()
class PTPadMetatype(PTOperatorMetatype):
    name = "PadOp"
    op_names = ["pad"]


@PT_OPERATOR_METATYPES.register()
class PTCatMetatype(PTOperatorMetatype):
    name = "CatOp"
    op_names = ["cat", "stack", "concat"]
    hw_config_names = [HWConfigOpName.CONCAT]


@PT_OPERATOR_METATYPES.register()
class PTRELUMetatype(PTOperatorMetatype):
    name = "ReluOp"
    op_names = ["relu", "relu_"]


@PT_OPERATOR_METATYPES.register()
class PTRELU6Metatype(PTOperatorMetatype):
    name = "Relu6Op"
    op_names = ["relu6"]


@PT_OPERATOR_METATYPES.register()
class PTMaxMetatype(PTOperatorMetatype):
    name = "MaxOp"
    op_names = ["max"]
    hw_config_names = [HWConfigOpName.MAXIMUM, HWConfigOpName.REDUCEMAX]


@PT_OPERATOR_METATYPES.register()
class PTMinMetatype(PTOperatorMetatype):
    name = "MinOp"
    op_names = ["min"]
    hw_config_names = [HWConfigOpName.MINIMUM]


@PT_OPERATOR_METATYPES.register()
class PTTransposeMetatype(PTOperatorMetatype):
    name = "TransposeOp"
    op_names = ["transpose", "permute", "transpose_"]
    hw_config_names = [HWConfigOpName.TRANSPOSE]


@PT_OPERATOR_METATYPES.register()
class PTGatherMetatype(PTOperatorMetatype):
    name = "GatherOp"
    op_names = ["index_select", "__getitem__", "gather", "select", "where"]


@PT_OPERATOR_METATYPES.register()
class PTScatterMetatype(PTOperatorMetatype):
    name = "ScatterOp"
    op_names = ["scatter", "masked_fill", "masked_fill_"]


@PT_OPERATOR_METATYPES.register()
class PTReshapeMetatype(PTOperatorMetatype):
    name = "ReshapeOp"
    op_names = ["reshape", "view", "flatten", "unflatten", "unsqueeze"]
    hw_config_names = [HWConfigOpName.RESHAPE, HWConfigOpName.UNSQUEEZE, HWConfigOpName.FLATTEN]


@PT_OPERATOR_METATYPES.register()
class PTSqueezeMetatype(PTOperatorMetatype):
    name = "SqueezeOp"
    op_names = ["squeeze"]
    hw_config_names = [HWConfigOpName.SQUEEZE]


@PT_OPERATOR_METATYPES.register()
class PTSplitMetatype(PTOperatorMetatype):
    name = "SplitOp"
    op_names = ["split", "chunk", "unbind"]
    hw_config_names = [HWConfigOpName.SPLIT, HWConfigOpName.CHUNK]


@PT_OPERATOR_METATYPES.register()
class PTExpandMetatype(PTOperatorMetatype):
    name = "ExpandOp"
    op_names = ["expand"]


@PT_OPERATOR_METATYPES.register()
class PTExpandAsMetatype(PTOperatorMetatype):
    name = "ExpandAsOp"
    op_names = ["expand_as"]


@PT_OPERATOR_METATYPES.register()
class PTEmbeddingMetatype(PTOperatorMetatype):
    name = "EmbeddingOp"
    op_names = ["embedding"]
    hw_config_names = [HWConfigOpName.EMBEDDING]
    weight_port_ids = [1]


@PT_OPERATOR_METATYPES.register()
class PTEmbeddingBagMetatype(PTOperatorMetatype):
    name = "EmbeddingBagOp"
    op_names = ["embedding_bag"]
    hw_config_names = [HWConfigOpName.EMBEDDINGBAG]
    weight_port_ids = [1]


@PT_OPERATOR_METATYPES.register()
class PTSoftmaxMetatype(PTOperatorMetatype):
    name = "SoftmaxOp"
    op_names = ["softmax"]


@PT_OPERATOR_METATYPES.register()
class PTLessMetatype(PTOperatorMetatype):
    name = "LessOp"
    op_names = ["__lt__"]
    hw_config_names = [HWConfigOpName.LESS]


@PT_OPERATOR_METATYPES.register()
class PTLessEqualMetatype(PTOperatorMetatype):
    name = "LessEqualOp"
    op_names = ["__le__"]
    hw_config_names = [HWConfigOpName.LESSEQUAL]


@PT_OPERATOR_METATYPES.register()
class PTGreaterMetatype(PTOperatorMetatype):
    name = "GreaterOp"
    op_names = ["__gt__", "gt"]
    hw_config_names = [HWConfigOpName.GREATER]


@PT_OPERATOR_METATYPES.register()
class PTGreaterEqualMetatype(PTOperatorMetatype):
    name = "GreaterEqualOp"
    op_names = ["__ge__", "ge"]
    hw_config_names = [HWConfigOpName.GREATEREQUAL]


@PT_OPERATOR_METATYPES.register()
class PTModMetatype(PTOperatorMetatype):
    name = "ModOp"
    op_names = ["__mod__"]
    hw_config_names = [HWConfigOpName.FLOORMOD]


@PT_OPERATOR_METATYPES.register()
class PTEqualsMetatype(PTOperatorMetatype):
    name = "EqualsOp"
    op_names = ["__eq__", "eq"]
    hw_config_names = [HWConfigOpName.EQUAL]


@PT_OPERATOR_METATYPES.register()
class PTNotEqualMetatype(PTOperatorMetatype):
    name = "NotEqualOp"
    op_names = ["__ne__", "ne"]
    hw_config_names = [HWConfigOpName.NOTEQUAL]


@PT_OPERATOR_METATYPES.register()
class PTLogicalOrMetatype(PTOperatorMetatype):
    name = "LogicalOrOp"
    op_names = ["__or__", "__ior__", "__ror__"]
    hw_config_names = [HWConfigOpName.LOGICALOR]


@PT_OPERATOR_METATYPES.register()
class PTLogicalXorMetatype(PTOperatorMetatype):
    name = "LogicalXorOp"
    op_names = ["__xor__", "__ixor__", "__rxor__"]
    hw_config_names = [HWConfigOpName.LOGICALXOR]


@PT_OPERATOR_METATYPES.register()
class PTLogicalAndMetatype(PTOperatorMetatype):
    name = "LogicalAndOp"
    op_names = ["__and__", "__iand__", "__rand__"]
    hw_config_names = [HWConfigOpName.LOGICALAND]


@PT_OPERATOR_METATYPES.register()
class PTLogicalNotMetatype(PTOperatorMetatype):
    name = "LogicalNotOp"
    op_names = ["logical_not_", "__invert__"]
    hw_config_names = [HWConfigOpName.LOGICALNOT]


@PT_OPERATOR_METATYPES.register()
class PTNegativeMetatype(PTOperatorMetatype):
    name = "NegativeOp"
    op_names = ["neg", "__neg__"]


@PT_OPERATOR_METATYPES.register()
class PTPowerMetatype(PTOperatorMetatype):
    name = "PowerOp"
    op_names = ["pow", "__pow__", "__ipow__", "__rpow__"]
    hw_config_names = [HWConfigOpName.POWER]


@PT_OPERATOR_METATYPES.register()
class PTSqrtMetatype(PTOperatorMetatype):
    name = "SqrtOp"
    op_names = ["sqrt", "sqrt_"]
    hw_config_names = [HWConfigOpName.POWER]


@PT_OPERATOR_METATYPES.register()
class PTInterpolateMetatype(PTOperatorMetatype):
    name = "InterpolateOp"
    op_names = ["interpolate"]
    hw_config_names = [HWConfigOpName.INTERPOLATE]
    num_expected_input_edges = 1


@PT_OPERATOR_METATYPES.register()
class PTRepeatMetatype(PTOperatorMetatype):
    name = "RepeatOp"
    op_names = ["repeat_interleave"]
    hw_config_names = [HWConfigOpName.TILE]
    num_expected_input_edges = 1


@PT_OPERATOR_METATYPES.register()
class PTPixelShuffleMetatype(PTOperatorMetatype):
    name = "PixelShuffleOp"
    op_names = ["pixel_shuffle"]
    num_expected_input_edges = 1


@PT_OPERATOR_METATYPES.register()
class PTSumMetatype(PTOperatorMetatype):
    name = "SumOp"
    op_names = ["sum"]
    hw_config_names = [HWConfigOpName.REDUCESUM]
    num_expected_input_edges = 1


@PT_OPERATOR_METATYPES.register()
class PTReduceL2(PTOperatorMetatype):
    name = "ReduceL2"
    op_names = ["normalize"]  # note: normalize is for general L_p normalization
    hw_config_names = [HWConfigOpName.REDUCEL2]
    num_expected_input_edges = 1


@PT_OPERATOR_METATYPES.register()
class PTScaledDotProductAttentionMetatype(PTOperatorMetatype):
    name = "ScaledDotProductAttentionOp"
    op_names = ["scaled_dot_product_attention"]
    hw_config_names = [HWConfigOpName.SCALED_DOT_PRODUCT_ATTENTION]
    target_input_ports = [0, 1]


def get_operator_metatypes() -> List[Type[OperatorMetatype]]:
    """
    Returns a list of the operator metatypes.

    :return: List of operator metatypes .
    """
    return list(PT_OPERATOR_METATYPES.registry_dict.values())
