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


'''
from typing import Dict, List, Optional, Type, TypeVar

from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.operator_metatypes import CONST_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import INPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OUTPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatypeRegistry
from nncf.common.hardware.opset import HWConfigOpName
from nncf.torch.dynamic_graph.graph import DynamicGraph
from nncf.torch.dynamic_graph.structs import NamespaceTarget

ModuleAttributes = TypeVar("ModuleAttributes", bound=BaseLayerAttributes)

FX_OPERATOR_METATYPES = OperatorMetatypeRegistry("operator_metatypes")

class FXOperatorMetatype(OperatorMetatype):
    """
    Base class for grouping PyTorch operators based on their semantic meaning.
    Each derived class represents a single semantic group - for example, AddMetatype would
    group together '__iadd__', '__add__' and '__radd__' operations which all define nodewise
    tensor addition.
    Derived classes also specify which PyTorch functions in which modules should be patched
    so that the entire group of operations is visible in the internal graph.
    Grouping also allows efficient application of HW specifics to compression of
    certain operation groups.

    :param external_op_names: Names of functions registered as operators via @register_operator to be associated
    with this metatype.
    :param module_to_function_names: Names of functions from 'torch.nn.function', 'torch.tensor' and 'torch' modules
    respectively, which are associated with this metatype.
    :param subtypes: List of subtypes of PyTorch operator.
    """

    classes: List[Type] = []

    subtypes: List[Type["FXOperatorMetatype"]] = []

    @classmethod
    def get_subtypes(cls) -> List[Type["FXOperatorMetatype"]]:
        return cls.subtypes.copy()

    @classmethod
    def determine_subtype(
        cls, layer_attributes: Optional[BaseLayerAttributes] = None, function_args=None, functions_kwargs=None
    ) -> Optional["FXOperatorSubtype"]:
        matches = []
        for subtype in cls.get_subtypes():
            if subtype.matches(layer_attributes, function_args, functions_kwargs):
                matches.append(subtype)
        assert len(matches) <= 1, "Multiple subtypes match operator call - cannot determine single subtype."
        if not matches:
            return None

        subtype = matches[0]
        nested_subtype = subtype.determine_subtype(layer_attributes, function_args, functions_kwargs)
        if nested_subtype:
            return nested_subtype
        return subtype


class FXOperatorSubtype:
    pass


class FXDepthwiseConvOperatorSubtype(FXOperatorSubtype):
    @classmethod
    def matches(
        cls, layer_attributes: Optional[BaseLayerAttributes] = None, function_args=None, functions_kwargs=None
    ) -> bool:
        if not isinstance(layer_attributes, ConvolutionLayerAttributes):
            return False
        if layer_attributes.groups == layer_attributes.in_channels and layer_attributes.in_channels > 1:
            return True
        return False


@FX_OPERATOR_METATYPES.register()
@INPUT_NOOP_METATYPES.register()
class FXPlaceholderMetatype(FXOperatorMetatype):
    name = "placeholder"
    external_op_names = [name, NNCFGraphNodeType.INPUT_NODE]


@FX_OPERATOR_METATYPES.register()
@OUTPUT_NOOP_METATYPES.register()
class FXOutputNoopMetatype(FXOperatorMetatype):
    name = "output_noop"
    external_op_names = [name, NNCFGraphNodeType.OUTPUT_NODE]


@FX_OPERATOR_METATYPES.register()
@CONST_NOOP_METATYPES.register()
class FXConstNoopMetatype(FXOperatorMetatype):
    name = "const_noop"
    external_op_names = [name, NNCFGraphNodeType.CONST_NODE]


@FX_OPERATOR_METATYPES.register()
@NOOP_METATYPES.register()
class FXNoopMetatype(FXOperatorMetatype):
    name = "noop"
    external_op_names = [name]
    module_to_function_names = {
        NamespaceTarget.TORCH_NN_FUNCTIONAL: [],
        NamespaceTarget.TORCH_TENSOR: ["contiguous", "clone", "detach", "detach_", "to"],
        NamespaceTarget.TORCH: ["clone", "detach", "detach_"],
    }


@FX_OPERATOR_METATYPES.register(is_subtype=True)
class FXDepthwiseConv1dSubtype(FXDepthwiseConvOperatorSubtype):
    name = "Conv1DOp"
    hw_config_name = [HWConfigOpName.DEPTHWISECONVOLUTION]
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["conv1d"]}
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@FX_OPERATOR_METATYPES.register()
class FXConv1dMetatype(FXOperatorMetatype):
    name = "Conv1DOp"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["conv1d"]}
    subtypes = [FXDepthwiseConv1dSubtype]
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@FX_OPERATOR_METATYPES.register(is_subtype=True)
class FXModuleDepthwiseConv2dSubtype(FXOperatorSubtype):
    name = "Conv2DOp"
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["conv2d"]}
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@FX_OPERATOR_METATYPES.register(is_subtype=True)
class FXDepthwiseConv2dSubtype(FXDepthwiseConvOperatorSubtype):
    name = "Conv2DOp"
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["conv2d"]}
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@FX_OPERATOR_METATYPES.register()
class FXConv2dMetatype(FXOperatorMetatype):
    name = "Conv2DOp"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["conv2d"]}
    subtypes = [FXDepthwiseConv2dSubtype]
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@FX_OPERATOR_METATYPES.register(is_subtype=True)
class FXModuleDepthwiseConv3dSubtype(FXModuleDepthwiseConvOperatorSubtype):
    name = "Conv3DOp"
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["conv3d"]}
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@FX_OPERATOR_METATYPES.register(is_subtype=True)
class FXDepthwiseConv3dSubtype(FXDepthwiseConvOperatorSubtype):
    name = "Conv3DOp"
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["conv3d"]}
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@FX_OPERATOR_METATYPES.register()
class FXConv3dMetatype(FXOperatorMetatype):
    name = "Conv3DOp"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["conv3d"]}
    subtypes = [FXDepthwiseConv3dSubtype]
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@FX_OPERATOR_METATYPES.register()
class FXConvTranspose1dMetatype(FXOperatorMetatype):
    name = "ConvTranspose1DOp"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["conv_transpose1d"]}
    subtypes = []
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@FX_OPERATOR_METATYPES.register()
class FXConvTranspose2dMetatype(FXOperatorMetatype):
    name = "ConvTranspose2DOp"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["conv_transpose2d"]}
    subtypes = []
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@FX_OPERATOR_METATYPES.register()
class FXConvTranspose3dMetatype(FXOperatorMetatype):
    name = "ConvTranspose3DOp"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["conv_transpose3d"]}
    subtypes = []
    output_channel_axis = 1
    num_expected_input_edges = 2
    weight_port_ids = [1]
    bias_port_id = 2


@FX_OPERATOR_METATYPES.register()
class FXDeformConv2dMetatype(FXOperatorMetatype):
    name = "DeformConv2dOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["deform_conv2d"]}
    subtypes = []
    num_expected_input_edges = 4
    weight_port_ids = [2]


@FX_OPERATOR_METATYPES.register()
class FXLinearMetatype(FXOperatorMetatype):
    name = "LinearOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["linear"]}
    hw_config_names = [HWConfigOpName.MATMUL]
    subtypes = []
    output_channel_axis = -1
    num_expected_input_edges = 2
    weight_port_ids = [1]


@FX_OPERATOR_METATYPES.register()
class FXHardTanhMetatype(FXOperatorMetatype):
    name = "HardTanhOP"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["hardtanh"]}
    num_expected_input_edges = 1


@FX_OPERATOR_METATYPES.register()
class FXHardSwishMetatype(FXOperatorMetatype):
    name = "HardSwishOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["hardswish"]}
    num_expected_input_edges = 1


@FX_OPERATOR_METATYPES.register()
class FXHardSigmoidMetatype(FXOperatorMetatype):
    name = "HardSigmoidOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["hardsigmoid"]}
    num_expected_input_edges = 1


@FX_OPERATOR_METATYPES.register()
class FXTanhMetatype(FXOperatorMetatype):
    name = "TanhOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["tanh"], NamespaceTarget.TORCH: ["tanh"]}
    num_expected_input_edges = 1


@FX_OPERATOR_METATYPES.register()
class FXELUMetatype(FXOperatorMetatype):
    name = "EluOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["elu", "elu_"]}
    num_expected_input_edges = 1


@FX_OPERATOR_METATYPES.register()
class FXPRELUMetatype(FXOperatorMetatype):
    name = "PReluOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["prelu"]}
    num_expected_input_edges = 1


@FX_OPERATOR_METATYPES.register()
class FXLeakyRELUMetatype(FXOperatorMetatype):
    name = "LeakyReluOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["leaky_relu"]}
    num_expected_input_edges = 1


@FX_OPERATOR_METATYPES.register(is_subtype=True)
@FX_OPERATOR_METATYPES.register()
class FXLayerNormMetatype(FXOperatorMetatype):
    name = "LayerNormOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["layer_norm"]}
    hw_config_names = [HWConfigOpName.MVN]
    subtypes = []
    num_expected_input_edges = 1
    weight_port_ids = [2]


@FX_OPERATOR_METATYPES.register()
class FXGroupNormMetatype(FXOperatorMetatype):
    name = "GroupNormOp"
    module_to_function_names = {}
    hw_config_names = [HWConfigOpName.MVN]
    subtypes = []
    weight_port_ids = [2]


@FX_OPERATOR_METATYPES.register()
class FXGELUMetatype(FXOperatorMetatype):
    name = "GeluOp"
    hw_config_names = [HWConfigOpName.GELU]
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["gelu"]}


@FX_OPERATOR_METATYPES.register()
class FXSILUMetatype(FXOperatorMetatype):
    name = "SiluOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["silu"]}


@FX_OPERATOR_METATYPES.register()
class FXSigmoidMetatype(FXOperatorMetatype):
    name = "SigmoidOp"
    module_to_function_names = {
        NamespaceTarget.TORCH_NN_FUNCTIONAL: ["sigmoid"],
        NamespaceTarget.TORCH_TENSOR: ["sigmoid"],
        NamespaceTarget.TORCH: ["sigmoid"],
    }


@FX_OPERATOR_METATYPES.register()
class FXAddMetatype(FXOperatorMetatype):
    name = "AddOp"
    module_to_function_names = {
        NamespaceTarget.TORCH_TENSOR: ["add", "__add__", "__iadd__", "__radd__"],
        NamespaceTarget.TORCH: ["add"],
    }
    hw_config_names = [HWConfigOpName.ADD]
    num_expected_input_edges = 2


@FX_OPERATOR_METATYPES.register()
class FXSubMetatype(FXOperatorMetatype):
    name = "SubOp"
    module_to_function_names = {
        NamespaceTarget.TORCH_TENSOR: ["sub", "__sub__", "__isub__", "__rsub__"],
        NamespaceTarget.TORCH: ["sub"],
    }
    hw_config_names = [HWConfigOpName.SUBTRACT]
    num_expected_input_edges = 2


@FX_OPERATOR_METATYPES.register()
class FXMulMetatype(FXOperatorMetatype):
    name = "MulOp"
    module_to_function_names = {
        NamespaceTarget.TORCH_TENSOR: ["mul", "__mul__", "__imul__", "__rmul__"],
        NamespaceTarget.TORCH: ["mul"],
    }
    hw_config_names = [HWConfigOpName.MULTIPLY]
    num_expected_input_edges = 2


@FX_OPERATOR_METATYPES.register()
class FXDivMetatype(FXOperatorMetatype):
    name = "DivOp"
    module_to_function_names = {
        NamespaceTarget.TORCH_TENSOR: [
            "div",
            "__div__",
            "__idiv__",
            "__rdiv__",
            "__truediv__",
            "__itruediv__",
            "__rtruediv__",
        ],
        NamespaceTarget.TORCH: ["div"],
    }
    hw_config_names = [HWConfigOpName.DIVIDE]
    num_expected_input_edges = 2


@FX_OPERATOR_METATYPES.register()
class FXFloorDivMetatype(FXOperatorMetatype):
    name = "FloordivOp"
    module_to_function_names = {
        NamespaceTarget.TORCH_TENSOR: ["__floordiv__", "__ifloordiv__", "__rfloordiv__"],
        NamespaceTarget.TORCH: ["floor_divide"],
    }
    num_expected_input_edges = 2


@FX_OPERATOR_METATYPES.register()
class FXExpMetatype(FXOperatorMetatype):
    name = "ExpOp"
    module_to_function_names = {
        NamespaceTarget.TORCH_TENSOR: ["exp"],
        NamespaceTarget.TORCH: ["exp"],
    }


@FX_OPERATOR_METATYPES.register()
class FXLogMetatype(FXOperatorMetatype):
    name = "LogOp"
    module_to_function_names = {
        NamespaceTarget.TORCH_TENSOR: ["log"],
        NamespaceTarget.TORCH: ["log"],
    }


@FX_OPERATOR_METATYPES.register()
class FXAbsMetatype(FXOperatorMetatype):
    name = "AbsOp"
    module_to_function_names = {
        NamespaceTarget.TORCH_TENSOR: ["abs", "__abs__"],
        NamespaceTarget.TORCH: ["abs"],
    }


@FX_OPERATOR_METATYPES.register()
class FXErfMetatype(FXOperatorMetatype):
    name = "ErfOp"
    module_to_function_names = {
        NamespaceTarget.TORCH: ["erf"],
    }


@FX_OPERATOR_METATYPES.register()
class FXMatMulMetatype(FXOperatorMetatype):
    name = "MatMulOp"
    module_to_function_names = {
        NamespaceTarget.TORCH_TENSOR: ["matmul", "__matmul__", "__rmatmul__"],
        NamespaceTarget.TORCH: ["matmul", "bmm", "mm"],
    }
    hw_config_names = [HWConfigOpName.MATMUL]
    num_expected_input_edges = 2
    weight_port_ids = [0, 1]


@FX_OPERATOR_METATYPES.register()
class FXAddmmMetatype(FXOperatorMetatype):
    name = "MatMulOp"
    module_to_function_names = {NamespaceTarget.TORCH: ["addmm", "baddbmm"]}
    hw_config_names = [HWConfigOpName.MATMUL]
    # 0-th arg to the baddbmm is basically a (b)ias to be (add)ed to the (bmm) operation,
    # presuming that most runtime implementations will fuse the bias addition into the matrix multiplication
    # and therefore won't quantize the bias input, as this would break the hardware-fused pattern.
    ignored_input_ports: List[int] = [0]
    num_expected_input_edges = 2
    weight_port_ids = [1, 2]


@FX_OPERATOR_METATYPES.register()
class FXMeanMetatype(FXOperatorMetatype):
    name = "MeanOp"
    module_to_function_names = {NamespaceTarget.TORCH_TENSOR: ["mean"]}
    hw_config_names = [HWConfigOpName.REDUCEMEAN]


@FX_OPERATOR_METATYPES.register()
class FXRoundMetatype(FXOperatorMetatype):
    name = "RoundOp"
    module_to_function_names = {NamespaceTarget.TORCH_TENSOR: ["round"]}


@FX_OPERATOR_METATYPES.register()
class FXDropoutMetatype(FXOperatorMetatype):
    name = "DropoutOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["dropout"]}


@FX_OPERATOR_METATYPES.register()
class FXThresholdMetatype(FXOperatorMetatype):
    name = "ThresholdOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["threshold"]}


@FX_OPERATOR_METATYPES.register()
class FXBatchNormMetatype(FXOperatorMetatype):
    name = "BatchNormOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["batch_norm"]}
    subtypes = []
    weight_port_ids = [3]
    bias_port_id = 4


@FX_OPERATOR_METATYPES.register()
class FXAvgPool2dMetatype(FXOperatorMetatype):
    name = "AvgPool2DOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["avg_pool2d", "adaptive_avg_pool2d"]}
    hw_config_names = [HWConfigOpName.AVGPOOL]


@FX_OPERATOR_METATYPES.register()
class FXAvgPool3dMetatype(FXOperatorMetatype):
    name = "AvgPool3DOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["avg_pool3d", "adaptive_avg_pool3d"]}
    hw_config_names = [HWConfigOpName.AVGPOOL]


class FXAdaptiveMaxPool1dMetatype(FXOperatorMetatype):
    name = "AdaptiveMaxPool1DOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["adaptive_max_pool1d"]}


@FX_OPERATOR_METATYPES.register()
class FXAdaptiveMaxPool2dMetatype(FXOperatorMetatype):
    name = "AdaptiveMaxPool2DOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["adaptive_max_pool2d"]}


@FX_OPERATOR_METATYPES.register()
class FXAdaptiveMaxPool3dMetatype(FXOperatorMetatype):
    name = "AdaptiveMaxPool3DOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["adaptive_max_pool3d"]}


class FXMaxPool1dMetatype(FXOperatorMetatype):
    name = "MaxPool1DOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["max_pool1d"]}
    hw_config_names = [HWConfigOpName.MAXPOOL]


@FX_OPERATOR_METATYPES.register()
class FXMaxPool2dMetatype(FXOperatorMetatype):
    name = "MaxPool2DOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["max_pool2d"]}
    hw_config_names = [HWConfigOpName.MAXPOOL]


@FX_OPERATOR_METATYPES.register()
class FXMaxPool3dMetatype(FXOperatorMetatype):
    name = "MaxPool3DOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["max_pool3d"]}
    hw_config_names = [HWConfigOpName.MAXPOOL]


@FX_OPERATOR_METATYPES.register()
class FXMaxUnpool1dMetatype(FXOperatorMetatype):
    name = "MaxUnPool1DOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["max_unpool1d"]}


@FX_OPERATOR_METATYPES.register()
class FXMaxUnpool2dMetatype(FXOperatorMetatype):
    name = "MaxUnPool2DOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["max_unpool2d"]}


@FX_OPERATOR_METATYPES.register()
class FXMaxUnpool3dMetatype(FXOperatorMetatype):
    name = "MaxUnPool3DOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["max_unpool3d"]}


@FX_OPERATOR_METATYPES.register()
class FXPadMetatype(FXOperatorMetatype):
    name = "PadOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["pad"]}


@FX_OPERATOR_METATYPES.register()
class FXCatMetatype(FXOperatorMetatype):
    name = "CatOp"
    module_to_function_names = {NamespaceTarget.TORCH: ["cat", "stack"]}
    hw_config_names = [HWConfigOpName.CONCAT]


@FX_OPERATOR_METATYPES.register()
class FXRELUMetatype(FXOperatorMetatype):
    name = "ReluOp"
    module_to_function_names = {NamespaceTarget.TORCH: ["relu", "relu_"]}


@FX_OPERATOR_METATYPES.register()
class FXRELU6Metatype(FXOperatorMetatype):
    name = "Relu6Op"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["relu6"]}


@FX_OPERATOR_METATYPES.register()
class FXMaxMetatype(FXOperatorMetatype):
    name = "MaxOp"
    module_to_function_names = {NamespaceTarget.TORCH: ["max"]}
    hw_config_names = [HWConfigOpName.MAXIMUM, HWConfigOpName.REDUCEMAX]


@FX_OPERATOR_METATYPES.register()
class FXMinMetatype(FXOperatorMetatype):
    name = "MinOp"
    module_to_function_names = {NamespaceTarget.TORCH: ["min"]}
    hw_config_names = [HWConfigOpName.MINIMUM]


@FX_OPERATOR_METATYPES.register()
class FXTransposeMetatype(FXOperatorMetatype):
    name = "TransposeOp"
    module_to_function_names = {
        NamespaceTarget.TORCH_TENSOR: ["transpose", "permute", "transpose_"],
        NamespaceTarget.TORCH: ["transpose"],
    }
    hw_config_names = [HWConfigOpName.TRANSPOSE]


@FX_OPERATOR_METATYPES.register()
class FXGatherMetatype(FXOperatorMetatype):
    name = "GatherOp"
    module_to_function_names = {
        NamespaceTarget.TORCH_TENSOR: ["index_select", "__getitem__"],
        NamespaceTarget.TORCH: ["gather", "index_select", "where"],
    }


@FX_OPERATOR_METATYPES.register()
class FXScatterMetatype(FXOperatorMetatype):
    name = "ScatterOp"
    module_to_function_names = {NamespaceTarget.TORCH_TENSOR: ["scatter", "masked_fill", "masked_fill_"]}


@FX_OPERATOR_METATYPES.register()
class FXReshapeMetatype(FXOperatorMetatype):
    name = "ReshapeOp"
    module_to_function_names = {
        NamespaceTarget.TORCH_TENSOR: ["reshape", "view", "flatten", "unsqueeze"],
        NamespaceTarget.TORCH: ["flatten", "unsqueeze"],
    }
    hw_config_names = [HWConfigOpName.RESHAPE, HWConfigOpName.UNSQUEEZE, HWConfigOpName.FLATTEN]


@FX_OPERATOR_METATYPES.register()
class FXSqueezeMetatype(FXOperatorMetatype):
    name = "SqueezeOp"
    module_to_function_names = {
        NamespaceTarget.TORCH_TENSOR: ["squeeze"],
        NamespaceTarget.TORCH: ["squeeze"],
    }
    hw_config_names = [HWConfigOpName.SQUEEZE]


@FX_OPERATOR_METATYPES.register()
class FXSplitMetatype(FXOperatorMetatype):
    name = "SplitOp"
    module_to_function_names = {
        NamespaceTarget.TORCH_NN_FUNCTIONAL: [],
        NamespaceTarget.TORCH_TENSOR: ["split", "chunk", "unbind"],
        NamespaceTarget.TORCH: ["split", "chunk", "unbind"],
    }
    hw_config_names = [HWConfigOpName.SPLIT, HWConfigOpName.CHUNK]


@FX_OPERATOR_METATYPES.register()
class FXExpandMetatype(FXOperatorMetatype):
    name = "ExpandOp"
    module_to_function_names = {NamespaceTarget.TORCH_TENSOR: ["expand"]}


@FX_OPERATOR_METATYPES.register()
class FXExpandAsMetatype(FXOperatorMetatype):
    name = "ExpandAsOp"
    module_to_function_names = {NamespaceTarget.TORCH_TENSOR: ["expand_as"]}


@FX_OPERATOR_METATYPES.register()
class FXEmbeddingMetatype(FXOperatorMetatype):
    name = "EmbeddingOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["embedding"]}
    hw_config_names = [HWConfigOpName.EMBEDDING]
    subtypes = []
    weight_port_ids = [1]


@FX_OPERATOR_METATYPES.register()
class FXEmbeddingBagMetatype(FXOperatorMetatype):
    name = "EmbeddingBagOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["embedding_bag"]}
    hw_config_names = [HWConfigOpName.EMBEDDINGBAG]
    subtypes = []
    weight_port_ids = [1]


@FX_OPERATOR_METATYPES.register()
class FXSoftmaxMetatype(FXOperatorMetatype):
    name = "SoftmaxOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["softmax"]}


@FX_OPERATOR_METATYPES.register()
class FXLessMetatype(FXOperatorMetatype):
    name = "LessOp"
    module_to_function_names = {NamespaceTarget.TORCH_TENSOR: ["__lt__"]}
    hw_config_names = [HWConfigOpName.LESS]


@FX_OPERATOR_METATYPES.register()
class FXLessEqualMetatype(FXOperatorMetatype):
    name = "LessEqualOp"
    module_to_function_names = {NamespaceTarget.TORCH_TENSOR: ["__le__"]}
    hw_config_names = [HWConfigOpName.LESSEQUAL]


@FX_OPERATOR_METATYPES.register()
class FXGreaterMetatype(FXOperatorMetatype):
    name = "GreaterOp"
    module_to_function_names = {NamespaceTarget.TORCH_TENSOR: ["__gt__"]}
    hw_config_names = [HWConfigOpName.GREATER]


@FX_OPERATOR_METATYPES.register()
class FXGreaterEqualMetatype(FXOperatorMetatype):
    name = "GreaterEqualOp"
    module_to_function_names = {NamespaceTarget.TORCH_TENSOR: ["__ge__"]}
    hw_config_names = [HWConfigOpName.GREATEREQUAL]


@FX_OPERATOR_METATYPES.register()
class FXModMetatype(FXOperatorMetatype):
    name = "ModOp"
    module_to_function_names = {NamespaceTarget.TORCH_TENSOR: ["__mod__"]}
    hw_config_names = [HWConfigOpName.FLOORMOD]


@FX_OPERATOR_METATYPES.register()
class FXEqualsMetatype(FXOperatorMetatype):
    name = "EqualsOp"
    module_to_function_names = {NamespaceTarget.TORCH_TENSOR: ["__eq__"]}
    hw_config_names = [HWConfigOpName.EQUAL]


@FX_OPERATOR_METATYPES.register()
class FXNotEqualMetatype(FXOperatorMetatype):
    name = "NotEqualOp"
    module_to_function_names = {NamespaceTarget.TORCH_TENSOR: ["__ne__"]}
    hw_config_names = [HWConfigOpName.NOTEQUAL]


@FX_OPERATOR_METATYPES.register()
class FXLogicalOrMetatype(FXOperatorMetatype):
    name = "LogicalOrOp"
    module_to_function_names = {NamespaceTarget.TORCH_TENSOR: ["__or__", "__ior__", "__ror__"]}
    hw_config_names = [HWConfigOpName.LOGICALOR]


@FX_OPERATOR_METATYPES.register()
class FXLogicalXorMetatype(FXOperatorMetatype):
    name = "LogicalXorOp"
    module_to_function_names = {NamespaceTarget.TORCH_TENSOR: ["__xor__", "__ixor__", "__rxor__"]}
    hw_config_names = [HWConfigOpName.LOGICALXOR]


@FX_OPERATOR_METATYPES.register()
class FXLogicalAndMetatype(FXOperatorMetatype):
    name = "LogicalAndOp"
    module_to_function_names = {NamespaceTarget.TORCH_TENSOR: ["__and__", "__iand__", "__rand__"]}
    hw_config_names = [HWConfigOpName.LOGICALAND]


@FX_OPERATOR_METATYPES.register()
class FXLogicalNotMetatype(FXOperatorMetatype):
    name = "LogicalNotOp"
    module_to_function_names = {NamespaceTarget.TORCH_TENSOR: ["logical_not_", "__invert__"]}
    hw_config_names = [HWConfigOpName.LOGICALNOT]


@FX_OPERATOR_METATYPES.register()
class FXNegativeMetatype(FXOperatorMetatype):
    name = "NegativeOp"
    module_to_function_names = {
        NamespaceTarget.TORCH_TENSOR: ["neg", "__neg__"],
        NamespaceTarget.TORCH: ["neg"],
    }


@FX_OPERATOR_METATYPES.register()
class FXPowerMetatype(FXOperatorMetatype):
    name = "PowerOp"
    module_to_function_names = {
        NamespaceTarget.TORCH_TENSOR: ["pow", "__pow__", "__ipow__", "__rpow__"],
        NamespaceTarget.TORCH: ["pow"],
    }
    hw_config_names = [HWConfigOpName.POWER]


@FX_OPERATOR_METATYPES.register()
class FXSqrtMetatype(FXOperatorMetatype):
    name = "SqrtOp"
    module_to_function_names = {
        NamespaceTarget.TORCH_TENSOR: ["sqrt", "sqrt_"],
        NamespaceTarget.TORCH: ["sqrt", "sqrt_"],
    }
    hw_config_names = [HWConfigOpName.POWER]


@FX_OPERATOR_METATYPES.register()
class FXInterpolateMetatype(FXOperatorMetatype):
    name = "InterpolateOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["interpolate"]}
    hw_config_names = [HWConfigOpName.INTERPOLATE]
    num_expected_input_edges = 1


@FX_OPERATOR_METATYPES.register()
class FXRepeatMetatype(FXOperatorMetatype):
    name = "RepeatOp"
    module_to_function_names = {NamespaceTarget.TORCH: ["repeat_interleave"]}
    hw_config_names = [HWConfigOpName.TILE]
    num_expected_input_edges = 1


@FX_OPERATOR_METATYPES.register()
class FXPixelShuffleMetatype(FXOperatorMetatype):
    name = "PixelShuffleOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["pixel_shuffle"]}
    num_expected_input_edges = 1


@FX_OPERATOR_METATYPES.register()
class FXSumMetatype(FXOperatorMetatype):
    name = "SumOp"
    module_to_function_names = {NamespaceTarget.TORCH_TENSOR: ["sum"], NamespaceTarget.TORCH: ["sum"]}
    hw_config_names = [HWConfigOpName.REDUCESUM]
    num_expected_input_edges = 1


@FX_OPERATOR_METATYPES.register()
class FXReduceL2(FXOperatorMetatype):
    name = "ReduceL2"
    module_to_function_names = {
        NamespaceTarget.TORCH_NN_FUNCTIONAL: ["normalize"],  # note: normalize is for general L_p normalization
    }
    hw_config_names = [HWConfigOpName.REDUCEL2]
    num_expected_input_edges = 1


def get_operator_metatypes() -> List[Type[OperatorMetatype]]:
    """
    Returns a list of the operator metatypes.

    :return: List of operator metatypes .
    """
    return list(FX_OPERATOR_METATYPES.registry_dict.values())


OPERATORS_WITH_WEIGHTS_METATYPES = [
    FXModuleConv1dMetatype,
    FXModuleConv2dMetatype,
    FXModuleConv3dMetatype,
    FXModuleDepthwiseConv1dSubtype,
    FXModuleDepthwiseConv2dSubtype,
    FXModuleDepthwiseConv3dSubtype,
    FXModuleLinearMetatype,
    FXModuleBatchNormMetatype,
    FXModuleGroupNormMetatype,
    FXModuleLayerNormMetatype,
    FXModuleConvTranspose1dMetatype,
    FXModuleConvTranspose2dMetatype,
    FXModuleConvTranspose3dMetatype,
    FXModuleEmbeddingMetatype,
    FXModuleEmbeddingBagMetatype,
]

UNIFICATION_PRODUCING_METATYPES = [
    FXModuleConv1dMetatype,
    FXModuleConv2dMetatype,
    FXModuleConv3dMetatype,
    FXModuleDepthwiseConv1dSubtype,
    FXModuleDepthwiseConv2dSubtype,
    FXModuleDepthwiseConv3dSubtype,
    FXModuleConvTranspose1dMetatype,
    FXModuleConvTranspose2dMetatype,
    FXModuleConvTranspose3dMetatype,
    FXModuleLinearMetatype,
]

OP_NAMES_WITH_WEIGHTS = [x for meta in OPERATORS_WITH_WEIGHTS_METATYPES for x in meta.get_all_aliases()]

QUANTIZE_NODE_TYPES = ["symmetric_quantize", "asymmetric_quantize"]

# These metatypes mix outputs for different samples into one axis.
# If reducers and aggregators collect statistics at the output of the following operations,
# assuming that 0-axis is batch axis, they get only 1 value instead of batch_size values.
# It could lead to inaccurate/incorrect statistics result.
OPERATIONS_OUTPUT_HAS_NO_BATCH_AXIS = [
    FXEmbeddingMetatype,
    FXEmbeddingBagMetatype,
    FXModuleEmbeddingBagMetatype,
    FXModuleEmbeddingMetatype,
]

'''
