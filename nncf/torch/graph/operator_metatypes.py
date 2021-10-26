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

from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar

from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.operator_metatypes import INPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OUTPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatypeRegistry
from nncf.common.hardware.opset import HWConfigOpName

ModuleAttributes = TypeVar('ModuleAttributes', bound=BaseLayerAttributes)

PT_OPERATOR_METATYPES = OperatorMetatypeRegistry("operator_metatypes")


class PTOperatorMetatype(OperatorMetatype):
    """
    Base class for grouping PyTorch operators based on their semantic meaning.
    Each derived class represents a single semantic group - for example, AddMetatype would
    group together '__iadd__', '__add__' and '__radd__' operations which all define nodewise
    tensor addition.
    Derived classes also specify which PyTorch functions in which modules should be patched
    and in what manner, so that the entire group of operations is visible in the internal graph
    representation. Grouping also allows efficient application of HW specifics to compression of
    certain operation groups.
    """
    op_names = []  # type: List[str]
    # Names of functions registered as operators via @register_operator to be associated
    # with this metatype
    external_op_names = []  # type: List[str]

    subtypes = []  # type: List[Type[OperatorMetatype]]

    @classmethod
    def get_subtypes(cls) -> List[Type[OperatorMetatype]]:
        return cls.subtypes

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return cls.op_names

    @classmethod
    def determine_subtype(cls,
                          layer_attributes: Optional[BaseLayerAttributes] = None,
                          function_args=None,
                          functions_kwargs=None) -> Optional['PTOperatorSubtype']:
        matches = []
        for subtype in cls.get_subtypes():
            if subtype.matches(layer_attributes,
                               function_args,
                               functions_kwargs):
                matches.append(subtype)
        assert len(matches) <= 1, "Multiple subtypes match operator call " \
                                  "- cannot determine single subtype."
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
    def matches(cls, layer_attributes: Optional[BaseLayerAttributes] = None,
                function_args=None,
                functions_kwargs=None) -> bool:
        raise NotImplementedError


@PT_OPERATOR_METATYPES.register()
@INPUT_NOOP_METATYPES.register()
class PTInputNoopMetatype(PTOperatorMetatype):
    name = "input_noop"
    external_op_names = [name, NNCFGraphNodeType.INPUT_NODE]

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return cls.external_op_names


@PT_OPERATOR_METATYPES.register()
@OUTPUT_NOOP_METATYPES.register()
class PTOutputNoopMetatype(PTOperatorMetatype):
    name = "output_noop"
    external_op_names = [name, NNCFGraphNodeType.OUTPUT_NODE]

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return cls.external_op_names


@PT_OPERATOR_METATYPES.register()
@NOOP_METATYPES.register()
class PTNoopMetatype(PTOperatorMetatype):
    name = "noop"
    external_op_names = [name]

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return cls.external_op_names


@PT_OPERATOR_METATYPES.register()
class PTDepthwiseConv1dSubtype(PTOperatorSubtype):
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]

    @classmethod
    def matches(cls, layer_attributes: Optional[ConvolutionLayerAttributes] = None,
                function_args=None,
                functions_kwargs=None) -> bool:
        if layer_attributes.groups == layer_attributes.in_channels and layer_attributes.in_channels > 1:
            return True
        return False


@PT_OPERATOR_METATYPES.register()
class PTConv1dMetatype(PTOperatorMetatype):
    name = "Conv1DOp"
    op_names = ["conv1d"]
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    subtypes = [PTDepthwiseConv1dSubtype]


@PT_OPERATOR_METATYPES.register()
class PTDepthwiseConv2dSubtype(PTOperatorSubtype):
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]

    @classmethod
    def matches(cls, layer_attributes: Optional[ConvolutionLayerAttributes] = None,
                function_args=None,
                functions_kwargs=None) -> bool:
        if layer_attributes.groups == layer_attributes.in_channels and layer_attributes.in_channels > 1:
            return True
        return False


@PT_OPERATOR_METATYPES.register()
class PTConv2dMetatype(PTOperatorMetatype):
    name = "Conv2DOp"
    op_names = ["conv2d"]
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    subtypes = [PTDepthwiseConv2dSubtype]


@PT_OPERATOR_METATYPES.register()
class PTDepthwiseConv3dSubtype(PTOperatorSubtype):
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]

    @classmethod
    def matches(cls, layer_attributes: Optional[ConvolutionLayerAttributes] = None,
                function_args=None,
                functions_kwargs=None) -> bool:
        if layer_attributes.groups == layer_attributes.in_channels and layer_attributes.in_channels > 1:
            return True
        return False


@PT_OPERATOR_METATYPES.register()
class PTConv3dMetatype(PTOperatorMetatype):
    name = "Conv3DOp"
    op_names = ["conv3d"]
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    subtypes = [PTDepthwiseConv3dSubtype]


@PT_OPERATOR_METATYPES.register()
class PTConvTranspose2dMetatype(PTOperatorMetatype):
    name = "ConvTranspose2DOp"
    op_names = ["conv_transpose2d"]
    hw_config_names = [HWConfigOpName.CONVOLUTION]


@PT_OPERATOR_METATYPES.register()
class PTConvTranspose3dMetatype(PTOperatorMetatype):
    name = "ConvTranspose3DOp"
    op_names = ["conv_transpose3d"]
    hw_config_names = [HWConfigOpName.CONVOLUTION]


@PT_OPERATOR_METATYPES.register()
class PTLinearMetatype(PTOperatorMetatype):
    name = "LinearOp"
    op_names = ["linear", "addmm"]
    hw_config_names = [HWConfigOpName.MATMUL]


@PT_OPERATOR_METATYPES.register()
class PTHardTanhMetatype(PTOperatorMetatype):
    name = "HardTanhOP"
    op_names = ["hardtanh"]


@PT_OPERATOR_METATYPES.register()
class PTHardSwishMetatype(PTOperatorMetatype):
    name = "HardSwishOp"
    op_names = ["hardswish"]


@PT_OPERATOR_METATYPES.register()
class PTTanhMetatype(PTOperatorMetatype):
    name = "TanhOp"
    op_names = ["tanh"]


@PT_OPERATOR_METATYPES.register()
class PTELUMetatype(PTOperatorMetatype):
    name = "EluOp"
    op_names = ["elu", "elu_"]


@PT_OPERATOR_METATYPES.register()
class PTPRELUMetatype(PTOperatorMetatype):
    name = "PReluOp"
    op_names = ["prelu"]


@PT_OPERATOR_METATYPES.register()
class PTLeakyRELUMetatype(PTOperatorMetatype):
    name = "LeakyReluOp"
    op_names = ["leaky_relu"]


@PT_OPERATOR_METATYPES.register()
class PTLayerNormMetatype(PTOperatorMetatype):
    name = "LayerNormOp"
    op_names = ["layer_norm"]
    hw_config_names = [HWConfigOpName.MVN]


@PT_OPERATOR_METATYPES.register()
class PTGroupNormMetatype(PTOperatorMetatype):
    name = "GroupNormOp"
    op_names = ["group_norm"]
    hw_config_names = [HWConfigOpName.MVN]


@PT_OPERATOR_METATYPES.register()
class PTGELUMetatype(PTOperatorMetatype):
    name = "GeluOp"
    op_names = ["gelu"]


@PT_OPERATOR_METATYPES.register()
class PTSILUMetatype(PTOperatorMetatype):
    name = "SiluOp"
    op_names = ["silu"]


@PT_OPERATOR_METATYPES.register()
class PTSigmoidMetatype(PTOperatorMetatype):
    name = "SigmoidOp"
    op_names = ["sigmoid"]


@PT_OPERATOR_METATYPES.register()
class PTAddMetatype(PTOperatorMetatype):
    name = "AddOp"
    op_names = ["add", "__add__", "__iadd__", "__radd__"]
    hw_config_names = [HWConfigOpName.ADD]


@PT_OPERATOR_METATYPES.register()
class PTSubMetatype(PTOperatorMetatype):
    name = "SubOp"
    op_names = ["sub", "__sub__", "__isub__", "__rsub__"]
    hw_config_names = [HWConfigOpName.SUBTRACT]


@PT_OPERATOR_METATYPES.register()
class PTMulMetatype(PTOperatorMetatype):
    name = "MulOp"
    op_names = ["mul", "__mul__", "__imul__", "__rmul__"]
    hw_config_names = [HWConfigOpName.MULTIPLY]


@PT_OPERATOR_METATYPES.register()
class PTDivMetatype(PTOperatorMetatype):
    name = "DivOp"
    op_names = ["div", "__div__", "__idiv__", "__truediv__"]
    hw_config_names = [HWConfigOpName.DIVIDE]


@PT_OPERATOR_METATYPES.register()
class PTFloorDivMetatype(PTOperatorMetatype):
    name = "FloordivOp"
    op_names = ["floordiv", "__floordiv__", "__ifloordiv__", "__rfloordiv__"]


@PT_OPERATOR_METATYPES.register()
class PTExpMetatype(PTOperatorMetatype):
    name = "ExpOp"
    op_names = ["exp"]


@PT_OPERATOR_METATYPES.register()
class PTErfMetatype(PTOperatorMetatype):
    name = "ErfOp"
    op_names = ["erf"]


@PT_OPERATOR_METATYPES.register()
class PTMatMulMetatype(PTOperatorMetatype):
    name = "MatMulOp"
    op_names = ["matmul", "bmm"]
    hw_config_names = [HWConfigOpName.MATMUL]


@PT_OPERATOR_METATYPES.register()
class PTMeanMetatype(PTOperatorMetatype):
    name = "MeanOp"
    op_names = ["mean"]
    hw_config_names = [HWConfigOpName.REDUCEMEAN]


@PT_OPERATOR_METATYPES.register()
class PTRoundMetatype(PTOperatorMetatype):
    name = "RoundOp"
    op_names = ["round"]


@PT_OPERATOR_METATYPES.register()
class PTDropoutMetatype(PTOperatorMetatype):
    name = "DropoutOp"
    op_names = ["dropout"]


@PT_OPERATOR_METATYPES.register()
class PTThresholdMetatype(PTOperatorMetatype):
    name = "ThresholdOp"
    op_names = ["threshold"]


@PT_OPERATOR_METATYPES.register()
class PTBatchNormMetatype(PTOperatorMetatype):
    name = "BatchNormOp"
    op_names = ["batch_norm"]


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


@PT_OPERATOR_METATYPES.register()
class PTMaxPool2dMetatype(PTOperatorMetatype):
    name = "MaxPool2DOp"
    op_names = ["max_pool2d"]
    hw_config_names = [HWConfigOpName.MAXPOOL]


@PT_OPERATOR_METATYPES.register()
class PTMaxPool3dMetatype(PTOperatorMetatype):
    name = "MaxPool3DOp"
    op_names = ["max_pool3d", "adaptive_max_pool3d"]
    hw_config_names = [HWConfigOpName.MAXPOOL]


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
    op_names = ["cat", "stack"]
    hw_config_names = [HWConfigOpName.CONCAT]


@PT_OPERATOR_METATYPES.register()
class PTRELUMetatype(PTOperatorMetatype):
    name = "ReluOp"
    op_names = ["relu", "relu_"]


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
    name = "TransposeOp"
    op_names = ["gather", "index_select", "where", "__getitem__"]


@PT_OPERATOR_METATYPES.register()
class PTScatterMetatype(PTOperatorMetatype):
    name = "ScatterOp"
    op_names = ["scatter", "masked_fill", "masked_fill_"]


@PT_OPERATOR_METATYPES.register()
class PTReshapeMetatype(PTOperatorMetatype):
    name = "ReshapeOp"
    op_names = ["reshape", "view", "squeeze", "flatten", "unsqueeze"]
    hw_config_names = [HWConfigOpName.RESHAPE, HWConfigOpName.SQUEEZE,
                       HWConfigOpName.UNSQUEEZE, HWConfigOpName.FLATTEN]


@PT_OPERATOR_METATYPES.register()
class PTSplitMetatype(PTOperatorMetatype):
    name = "SplitOp"
    op_names = ["split", "chunk"]
    hw_config_names = [HWConfigOpName.SPLIT]


@PT_OPERATOR_METATYPES.register()
class PTExpandMetatype(PTOperatorMetatype):
    name = "ExpandOp"
    op_names = ["expand"]


# Non-quantizable ops
@PT_OPERATOR_METATYPES.register()
class PTEmbeddingMetatype(PTOperatorMetatype):
    name = "EmbeddingOp"
    op_names = ["embedding"]
    hw_config_names = [HWConfigOpName.EMBEDDING]


@PT_OPERATOR_METATYPES.register()
class PTEmbeddingBagMetatype(PTOperatorMetatype):
    name = "EmbeddingBagOp"
    op_names = ["embedding_bag"]
    hw_config_names = [HWConfigOpName.EMBEDDINGBAG]


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
    op_names = ["__gt__"]
    hw_config_names = [HWConfigOpName.GREATER]


@PT_OPERATOR_METATYPES.register()
class PTGreaterEqualMetatype(PTOperatorMetatype):
    name = "GreaterEqualOp"
    op_names = ["__ge__"]
    hw_config_names = [HWConfigOpName.GREATEREQUAL]


@PT_OPERATOR_METATYPES.register()
class PTModMetatype(PTOperatorMetatype):
    name = "ModOp"
    op_names = ["__mod__"]
    hw_config_names = [HWConfigOpName.FLOORMOD]


@PT_OPERATOR_METATYPES.register()
class PTEqualsMetatype(PTOperatorMetatype):
    name = "EqualsOp"
    op_names = ["__eq__"]
    hw_config_names = [HWConfigOpName.EQUAL]


@PT_OPERATOR_METATYPES.register()
class PTNotEqualMetatype(PTOperatorMetatype):
    name = "NotEqualOp"
    op_names = ["__ne__"]
    hw_config_names = [HWConfigOpName.NOTEQUAL]


@PT_OPERATOR_METATYPES.register()
class PTLogicalOrMetatype(PTOperatorMetatype):
    name = "LogicalOrOp"
    op_names = ["__or__"]
    hw_config_names = [HWConfigOpName.LOGICALOR]


@PT_OPERATOR_METATYPES.register()
class PTLogicalXorMetatype(PTOperatorMetatype):
    name = "LogicalXorOp"
    op_names = ["__xor__"]
    hw_config_names = [HWConfigOpName.LOGICALXOR]


@PT_OPERATOR_METATYPES.register()
class PTLogicalAndMetatype(PTOperatorMetatype):
    name = "LogicalAndOp"
    op_names = ["__and__"]
    hw_config_names = [HWConfigOpName.LOGICALAND]


@PT_OPERATOR_METATYPES.register()
class PTLogicalNotMetatype(PTOperatorMetatype):
    name = "LogicalNotOp"
    op_names = ["logical_not_"]
    hw_config_names = [HWConfigOpName.LOGICALNOT]


@PT_OPERATOR_METATYPES.register()
class PTPowerMetatype(PTOperatorMetatype):
    name = "PowerOp"
    op_names = ["__pow__", "pow", "sqrt"]
    hw_config_names = [HWConfigOpName.POWER]


@PT_OPERATOR_METATYPES.register()
class PTInterpolateMetatype(PTOperatorMetatype):
    name = "InterpolateOp"
    op_names = ["interpolate"]
    hw_config_names = [HWConfigOpName.INTERPOLATE]


@PT_OPERATOR_METATYPES.register()
class PTRepeatMetatype(PTOperatorMetatype):
    name = "RepeatOp"
    op_names = ["repeat_interleave"]
    hw_config_names = [HWConfigOpName.TILE]


@PT_OPERATOR_METATYPES.register()
class PTPixelShuffleMetatype(PTOperatorMetatype):
    name = "PixelShuffleOp"
    op_names = ["pixel_shuffle"]


@PT_OPERATOR_METATYPES.register()
class PTSumMetatype(OperatorMetatype):
    name = "SumOp"
    op_names = ["sum"]
    hw_config_names = [HWConfigOpName.REDUCESUM]


def get_operator_metatypes() -> List[Type[OperatorMetatype]]:
    """
    Returns a list of the operator metatypes.

    :return: List of operator metatypes .
    """
    return list(PT_OPERATOR_METATYPES.registry_dict.values())
