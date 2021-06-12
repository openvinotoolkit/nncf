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

from copy import copy
from typing import List, Optional, Type
from typing import TypeVar

from nncf.common.graph import NNCFGraphNodeType
from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatypeRegistry
from nncf.common.hardware.opset import HWConfigOpName
from nncf.torch.dynamic_graph.trace_functions import CustomTraceFunction
from nncf.torch.dynamic_graph.trace_functions import ForwardTraceOnly


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
    # Wrapping specifications for operator calls of the following kind:
    # torch.nn.functional.conv2d
    torch_nn_functional_patch_spec = None  # type: Optional[PTPatchSpec]

    # Wrapping specifications for operator calls of the following kind:
    # torch.cat
    torch_module_patch_spec = None  # type: Optional[PTPatchSpec]

    # Wrapping specifications for operator calls of the following kind:
    # x = torch.Tensor(...)
    # x1 = x.view(...)
    torch_tensor_patch_spec = None  # type: Optional[PTPatchSpec]

    # Names of functions registered as operators via @register_operator to be associated
    # with this metatype
    external_op_names = []  # type: List[str]

    subtypes = []  # type: List[Type[OperatorMetatype]]

    @classmethod
    def get_subtypes(cls) -> List[Type[OperatorMetatype]]:
        return cls.subtypes

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        # TODO: disambiguate overlapping function names
        retval = copy(cls.external_op_names)
        if cls.torch_nn_functional_patch_spec is not None:
            for fn_name in cls.torch_nn_functional_patch_spec.underlying_function_names:
                retval.append(fn_name)
        if cls.torch_module_patch_spec is not None:
            for fn_name in cls.torch_module_patch_spec.underlying_function_names:
                retval.append(fn_name)
        if cls.torch_tensor_patch_spec is not None:
            for fn_name in cls.torch_tensor_patch_spec.underlying_function_names:
                retval.append(fn_name)
        return retval

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


class PTPatchSpec:
    def __init__(self,
                 underlying_function_names: List[str],
                 custom_trace_fn: CustomTraceFunction = None):
        """
        :param underlying_function_names: All function names in this list will be wrapped with NNCF
        wrappers that allow corresponding function calls to be registered in NNCF internal graph
        representation of the PyTorch model and to be afterwards considered for compression.
        :param custom_trace_fn: Will be called instead of the regular node search/insertion step
        during the corresponding operator call. Useful to specify this for nodes that have no effect on compression
        and therefore not vital to graph representation, but that should still be accounted for so that the
        graph representation does not become disjoint."""
        self.underlying_function_names = underlying_function_names
        self.custom_trace_fn = custom_trace_fn


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
class InputNoopMetatype(PTOperatorMetatype):
    name = "input_noop"
    external_op_names = [name, NNCFGraphNodeType.INPUT_NODE]


@PT_OPERATOR_METATYPES.register()
class OutputNoopMetatype(PTOperatorMetatype):
    name = "output_noop"
    external_op_names = [name, NNCFGraphNodeType.OUTPUT_NODE]


@PT_OPERATOR_METATYPES.register()
class NoopMetatype(PTOperatorMetatype):
    name = "noop"
    external_op_names = [name]


@PT_OPERATOR_METATYPES.register()
class DepthwiseConv1dSubtype(PTOperatorSubtype):
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]

    @classmethod
    def matches(cls, layer_attributes: Optional[ConvolutionLayerAttributes] = None,
                function_args=None,
                functions_kwargs=None) -> bool:
        if layer_attributes.groups == layer_attributes.in_channels and layer_attributes.in_channels > 1:
            return True
        return False


@PT_OPERATOR_METATYPES.register()
class Conv1dMetatype(PTOperatorMetatype):
    name = "conv1d"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    subtypes = [DepthwiseConv1dSubtype]
    torch_nn_functional_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class DepthwiseConv2dSubtype(PTOperatorSubtype):
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]

    @classmethod
    def matches(cls, layer_attributes: Optional[ConvolutionLayerAttributes] = None,
                function_args=None,
                functions_kwargs=None) -> bool:
        if layer_attributes.groups == layer_attributes.in_channels and layer_attributes.in_channels > 1:
            return True
        return False


@PT_OPERATOR_METATYPES.register()
class Conv2dMetatype(PTOperatorMetatype):
    name = "conv2d"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    torch_nn_functional_patch_spec = PTPatchSpec([name])
    subtypes = [DepthwiseConv2dSubtype]


@PT_OPERATOR_METATYPES.register()
class DepthwiseConv3dSubtype(PTOperatorSubtype):
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]

    @classmethod
    def matches(cls, layer_attributes: Optional[ConvolutionLayerAttributes] = None,
                function_args=None,
                functions_kwargs=None) -> bool:
        if layer_attributes.groups == layer_attributes.in_channels and layer_attributes.in_channels > 1:
            return True
        return False


@PT_OPERATOR_METATYPES.register()
class Conv3dMetatype(PTOperatorMetatype):
    name = "conv3d"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    subtypes = [DepthwiseConv3dSubtype]
    torch_nn_functional_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class ConvTranspose2dMetatype(PTOperatorMetatype):
    name = "conv_transpose2d"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    torch_nn_functional_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class ConvTranspose3dMetatype(PTOperatorMetatype):
    name = "conv_transpose3d"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    torch_nn_functional_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class LinearMetatype(PTOperatorMetatype):
    name = "linear"
    hw_config_names = [HWConfigOpName.MATMUL]
    torch_nn_functional_patch_spec = PTPatchSpec([name])
    torch_module_patch_spec = PTPatchSpec(["addmm"])


@PT_OPERATOR_METATYPES.register()
class HardTanhMetatype(PTOperatorMetatype):
    name = "hardtanh"
    torch_nn_functional_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class TanhMetatype(PTOperatorMetatype):
    name = "tanh"
    torch_nn_functional_patch_spec = PTPatchSpec([name])
    torch_module_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class ELUMetatype(PTOperatorMetatype):
    name = "elu"
    torch_nn_functional_patch_spec = PTPatchSpec([name, "elu_"])


@PT_OPERATOR_METATYPES.register()
class PRELUMetatype(PTOperatorMetatype):
    name = "prelu"
    torch_nn_functional_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class LeakyRELUMetatype(PTOperatorMetatype):
    name = "leaky_relu"
    torch_nn_functional_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class LayerNormMetatype(PTOperatorMetatype):
    name = "layer_norm"
    torch_nn_functional_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.MVN]

@PT_OPERATOR_METATYPES.register()
class GroupNormMetatype(PTOperatorMetatype):
    name = "group_norm"
    torch_nn_functional_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class GELUMetatype(PTOperatorMetatype):
    name = "gelu"
    torch_nn_functional_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class SigmoidMetatype(PTOperatorMetatype):
    name = "sigmoid"
    torch_nn_functional_patch_spec = PTPatchSpec([name])
    torch_module_patch_spec = PTPatchSpec([name])
    torch_tensor_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class AddMetatype(PTOperatorMetatype):
    name = "add"
    torch_tensor_patch_spec = PTPatchSpec([name,
                                         "__add__",
                                         "__iadd__",
                                         "__radd__"])
    torch_module_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.ADD]


@PT_OPERATOR_METATYPES.register()
class SubMetatype(PTOperatorMetatype):
    name = "sub"
    torch_tensor_patch_spec = PTPatchSpec(["__sub__",
                                         "__isub__",
                                         "__rsub__"])
    hw_config_names = [HWConfigOpName.SUBTRACT]


@PT_OPERATOR_METATYPES.register()
class MulMetatype(PTOperatorMetatype):
    name = "mul"
    torch_tensor_patch_spec = PTPatchSpec(["mul",
                                         "__mul__",
                                         "__imul__",
                                         "__rmul__"])
    hw_config_names = [HWConfigOpName.MULTIPLY]


@PT_OPERATOR_METATYPES.register()
class DivMetatype(PTOperatorMetatype):
    name = "div"
    torch_module_patch_spec = PTPatchSpec([name])
    torch_tensor_patch_spec = PTPatchSpec(["__div__",
                                         "__idiv__",
                                         "__truediv__"])
    hw_config_names = [HWConfigOpName.DIVIDE]


@PT_OPERATOR_METATYPES.register()
class ExpMetatype(PTOperatorMetatype):
    name = "exp"
    torch_module_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class ErfMetatype(PTOperatorMetatype):
    name = "erf"
    torch_module_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class MatMulMetatype(PTOperatorMetatype):
    name = "matmul"
    torch_module_patch_spec = PTPatchSpec([name, "bmm"])
    torch_tensor_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.MATMUL]


@PT_OPERATOR_METATYPES.register()
class MeanMetatype(PTOperatorMetatype):
    name = "mean"
    torch_tensor_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.REDUCEMEAN]


@PT_OPERATOR_METATYPES.register()
class RoundMetatype(PTOperatorMetatype):
    name = "round"
    torch_tensor_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class DropoutMetatype(PTOperatorMetatype):
    name = "dropout"
    torch_nn_functional_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class ThresholdMetatype(PTOperatorMetatype):
    name = "threshold"
    torch_nn_functional_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class BatchNormMetatype(PTOperatorMetatype):
    name = "batch_norm"
    torch_nn_functional_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class AvgPool2dMetatype(PTOperatorMetatype):
    name = "avg_pool2d"
    hw_config_names = [HWConfigOpName.AVGPOOL]
    torch_nn_functional_patch_spec = PTPatchSpec([name, "adaptive_avg_pool2d"])


@PT_OPERATOR_METATYPES.register()
class AvgPool3dMetatype(PTOperatorMetatype):
    name = "avg_pool3d"
    hw_config_names = [HWConfigOpName.AVGPOOL]
    torch_nn_functional_patch_spec = PTPatchSpec([name, "adaptive_avg_pool3d"])


@PT_OPERATOR_METATYPES.register()
class MaxPool2dMetatype(PTOperatorMetatype):
    name = "max_pool2d"
    torch_nn_functional_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.MAXPOOL]


@PT_OPERATOR_METATYPES.register()
class MaxPool3dMetatype(PTOperatorMetatype):
    name = "max_pool3d"
    torch_nn_functional_patch_spec = PTPatchSpec([name, "adaptive_max_pool3d"])
    hw_config_names = [HWConfigOpName.MAXPOOL]


@PT_OPERATOR_METATYPES.register()
class MaxUnpool3dMetatype(PTOperatorMetatype):
    name = "max_unpool3d"
    torch_nn_functional_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class PadMetatype(PTOperatorMetatype):
    name = "pad"
    torch_nn_functional_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class CatMetatype(PTOperatorMetatype):
    name = "cat"
    torch_module_patch_spec = PTPatchSpec([name, "stack"])
    hw_config_names = [HWConfigOpName.CONCAT]


@PT_OPERATOR_METATYPES.register()
class RELUMetatype(PTOperatorMetatype):
    name = "relu"
    torch_module_patch_spec = PTPatchSpec([name, "relu_"])


@PT_OPERATOR_METATYPES.register()
class MaxMetatype(PTOperatorMetatype):
    name = "max"
    torch_module_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.MAXIMUM,
                       HWConfigOpName.REDUCEMAX]


@PT_OPERATOR_METATYPES.register()
class MinMetatype(PTOperatorMetatype):
    name = "min"
    torch_module_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.MINIMUM]


@PT_OPERATOR_METATYPES.register()
class ARangeMetatype(PTOperatorMetatype):
    name = "arange"
    torch_module_patch_spec = PTPatchSpec([name], ForwardTraceOnly())


@PT_OPERATOR_METATYPES.register()
class TransposeMetatype(PTOperatorMetatype):
    name = "transpose"
    torch_module_patch_spec = PTPatchSpec([name])
    torch_tensor_patch_spec = PTPatchSpec([name, "permute"])
    hw_config_names = [HWConfigOpName.TRANSPOSE]


@PT_OPERATOR_METATYPES.register()
class GatherMetatype(PTOperatorMetatype):
    name = "gather"
    torch_module_patch_spec = PTPatchSpec(["index_select", "where"])
    torch_tensor_patch_spec = PTPatchSpec(["index_select", "__getitem__"])


@PT_OPERATOR_METATYPES.register()
class ScatterMetatype(PTOperatorMetatype):
    name = "scatter"
    torch_tensor_patch_spec = PTPatchSpec(["masked_fill", "masked_fill_"])


@PT_OPERATOR_METATYPES.register()
class ReshapeMetatype(PTOperatorMetatype):
    name = "reshape"
    torch_module_patch_spec = PTPatchSpec(["squeeze", "flatten", "unsqueeze"])
    torch_tensor_patch_spec = PTPatchSpec([name, "view", "flatten", "squeeze", "unsqueeze"])
    hw_config_names = [HWConfigOpName.RESHAPE,
                       HWConfigOpName.SQUEEZE,
                       HWConfigOpName.UNSQUEEZE,
                       HWConfigOpName.FLATTEN]


@PT_OPERATOR_METATYPES.register()
class ContiguousMetatype(PTOperatorMetatype):
    name = "contiguous"
    torch_tensor_patch_spec = PTPatchSpec([name], ForwardTraceOnly())


@PT_OPERATOR_METATYPES.register()
class SplitMetatype(PTOperatorMetatype):
    name = "split"
    torch_tensor_patch_spec = PTPatchSpec([name, "chunk"])
    hw_config_names = [HWConfigOpName.SPLIT]


@PT_OPERATOR_METATYPES.register()
class ExpandMetatype(PTOperatorMetatype):
    name = "expand"
    torch_tensor_patch_spec = PTPatchSpec([name])


# Non-quantizable ops
@PT_OPERATOR_METATYPES.register()
class EmbeddingMetatype(PTOperatorMetatype):
    name = "embedding"
    torch_nn_functional_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.EMBEDDING]


@PT_OPERATOR_METATYPES.register()
class EmbeddingBagMetatype(PTOperatorMetatype):
    name = "embedding_bag"
    torch_nn_functional_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.EMBEDDINGBAG]


@PT_OPERATOR_METATYPES.register()
class SoftmaxMetatype(PTOperatorMetatype):
    name = "softmax"
    torch_nn_functional_patch_spec = PTPatchSpec([name])


@PT_OPERATOR_METATYPES.register()
class LessMetatype(PTOperatorMetatype):
    name = "__lt__"
    torch_tensor_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.LESS]


@PT_OPERATOR_METATYPES.register()
class LessEqualMetatype(PTOperatorMetatype):
    name = "__le__"
    torch_tensor_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.LESSEQUAL]


@PT_OPERATOR_METATYPES.register()
class GreaterMetatype(PTOperatorMetatype):
    name = "__gt__"
    torch_tensor_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.GREATER]


@PT_OPERATOR_METATYPES.register()
class GreaterEqualMetatype(PTOperatorMetatype):
    name = "__ge__"
    torch_tensor_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.GREATEREQUAL]


@PT_OPERATOR_METATYPES.register()
class ModMetatype(PTOperatorMetatype):
    name = "__mod__"
    torch_tensor_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.FLOORMOD]


@PT_OPERATOR_METATYPES.register()
class EqualsMetatype(PTOperatorMetatype):
    name = "__eq__"
    torch_tensor_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.EQUAL]


@PT_OPERATOR_METATYPES.register()
class NotEqualMetatype(PTOperatorMetatype):
    name = "__ne__"
    torch_tensor_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.NOTEQUAL]


@PT_OPERATOR_METATYPES.register()
class LogicalOrMetatype(PTOperatorMetatype):
    name = "__or__"
    torch_tensor_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.LOGICALOR]


@PT_OPERATOR_METATYPES.register()
class LogicalXorMetatype(PTOperatorMetatype):
    name = "__xor__"
    torch_tensor_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.LOGICALXOR]


@PT_OPERATOR_METATYPES.register()
class LogicalAndMetatype(PTOperatorMetatype):
    name = "__and__"
    torch_tensor_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.LOGICALAND]


@PT_OPERATOR_METATYPES.register()
class LogicalNotMetatype(PTOperatorMetatype):
    name = "logical_not_"
    torch_tensor_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.LOGICALNOT]


@PT_OPERATOR_METATYPES.register()
class PowerMetatype(PTOperatorMetatype):
    name = "__pow__"
    torch_tensor_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.POWER]


@PT_OPERATOR_METATYPES.register()
class InterpolateMetatype(PTOperatorMetatype):
    name = "interpolate"
    torch_nn_functional_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.INTERPOLATE]


@PT_OPERATOR_METATYPES.register()
class RepeatMetatype(PTOperatorMetatype):
    name = "repeat_interleave"
    torch_module_patch_spec = PTPatchSpec([name])
    hw_config_names = [HWConfigOpName.TILE]


@PT_OPERATOR_METATYPES.register()
class CloneMetatype(PTOperatorMetatype):
    name = "clone"
    torch_tensor_patch_spec = PTPatchSpec([name], ForwardTraceOnly())


@PT_OPERATOR_METATYPES.register()
class PixelShuffleMetatype(PTOperatorMetatype):
    name = "pixel_shuffle"
    torch_nn_functional_patch_spec = PTPatchSpec([name])


def get_operator_metatypes() -> List[Type[OperatorMetatype]]:
    """
    Returns a list of the operator metatypes.

    :return: List of operator metatypes .
    """
    return list(PT_OPERATOR_METATYPES.registry_dict.values())


def get_input_metatypes() -> List[Type[OperatorMetatype]]:
    """
    Returns a list of the input operator metatypes.

    :return: List of the input operator metatypes .
    """
    return [InputNoopMetatype]


def get_output_metatypes() -> List[Type[OperatorMetatype]]:
    """
    Returns a list of the output operator metatypes.

    :return: List of the output operator metatypes .
    """
    return [OutputNoopMetatype]
