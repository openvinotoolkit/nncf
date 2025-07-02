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

from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.experimental.common.pruning.operations import BatchNormPruningOp
from nncf.experimental.common.pruning.operations import ElementwisePruningOp
from nncf.experimental.common.pruning.operations import ExpandAsPruningOp
from nncf.experimental.common.pruning.operations import GatherPruningOp
from nncf.experimental.common.pruning.operations import GroupNormPruningOp
from nncf.experimental.common.pruning.operations import IdentityMaskForwardPruningOp
from nncf.experimental.common.pruning.operations import InputPruningOp
from nncf.experimental.common.pruning.operations import LinearPruningOp
from nncf.experimental.common.pruning.operations import OutputPruningOp
from nncf.experimental.common.pruning.operations import ReshapePruningOp
from nncf.experimental.common.pruning.operations import ScatterPruningOp
from nncf.experimental.common.pruning.operations import SplitPruningOp
from nncf.experimental.common.pruning.operations import StopMaskForwardPruningOp
from nncf.experimental.common.pruning.operations import TransposePruningOp
from nncf.torch.graph.operator_metatypes import PTAdaptiveMaxPool2dMetatype
from nncf.torch.graph.operator_metatypes import PTAddMetatype
from nncf.torch.graph.operator_metatypes import PTAvgPool2dMetatype
from nncf.torch.graph.operator_metatypes import PTBatchNormMetatype
from nncf.torch.graph.operator_metatypes import PTDivMetatype
from nncf.torch.graph.operator_metatypes import PTDropoutMetatype
from nncf.torch.graph.operator_metatypes import PTELUMetatype
from nncf.torch.graph.operator_metatypes import PTExpandAsMetatype
from nncf.torch.graph.operator_metatypes import PTGatherMetatype
from nncf.torch.graph.operator_metatypes import PTGELUMetatype
from nncf.torch.graph.operator_metatypes import PTGroupNormMetatype
from nncf.torch.graph.operator_metatypes import PTHardSigmoidMetatype
from nncf.torch.graph.operator_metatypes import PTHardSwishMetatype
from nncf.torch.graph.operator_metatypes import PTHardTanhMetatype
from nncf.torch.graph.operator_metatypes import PTInputNoopMetatype
from nncf.torch.graph.operator_metatypes import PTLeakyRELUMetatype
from nncf.torch.graph.operator_metatypes import PTLinearMetatype
from nncf.torch.graph.operator_metatypes import PTMatMulMetatype
from nncf.torch.graph.operator_metatypes import PTMaxMetatype
from nncf.torch.graph.operator_metatypes import PTMaxPool2dMetatype
from nncf.torch.graph.operator_metatypes import PTMeanMetatype
from nncf.torch.graph.operator_metatypes import PTMinMetatype
from nncf.torch.graph.operator_metatypes import PTMulMetatype
from nncf.torch.graph.operator_metatypes import PTNoopMetatype
from nncf.torch.graph.operator_metatypes import PTOutputNoopMetatype
from nncf.torch.graph.operator_metatypes import PTPowerMetatype
from nncf.torch.graph.operator_metatypes import PTPRELUMetatype
from nncf.torch.graph.operator_metatypes import PTRELU6Metatype
from nncf.torch.graph.operator_metatypes import PTRELUMetatype
from nncf.torch.graph.operator_metatypes import PTReshapeMetatype
from nncf.torch.graph.operator_metatypes import PTScatterMetatype
from nncf.torch.graph.operator_metatypes import PTSigmoidMetatype
from nncf.torch.graph.operator_metatypes import PTSILUMetatype
from nncf.torch.graph.operator_metatypes import PTSoftmaxMetatype
from nncf.torch.graph.operator_metatypes import PTSplitMetatype
from nncf.torch.graph.operator_metatypes import PTSqueezeMetatype
from nncf.torch.graph.operator_metatypes import PTSubMetatype
from nncf.torch.graph.operator_metatypes import PTSumMetatype
from nncf.torch.graph.operator_metatypes import PTTanhMetatype
from nncf.torch.graph.operator_metatypes import PTTransposeMetatype

PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES = PruningOperationsMetatypeRegistry("operator_metatypes")


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register("model_input")
class PTInputPruningOp(InputPruningOp):
    subtypes = [PTInputNoopMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register("model_output")
class PTOutputPruningOp(OutputPruningOp):
    subtypes = [PTOutputNoopMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register("identity_mask_propagation")
class PTIdentityMaskForwardPruningOp(IdentityMaskForwardPruningOp):
    subtypes = [
        PTHardTanhMetatype,
        PTTanhMetatype,
        PTRELUMetatype,
        PTRELU6Metatype,
        PTLeakyRELUMetatype,
        PTPRELUMetatype,
        PTELUMetatype,
        PTGELUMetatype,
        PTSigmoidMetatype,
        PTSoftmaxMetatype,
        PTAvgPool2dMetatype,
        PTAdaptiveMaxPool2dMetatype,
        PTMaxPool2dMetatype,
        PTMeanMetatype,
        PTDropoutMetatype,
        PTSILUMetatype,
        PTPowerMetatype,
        PTHardSwishMetatype,
        PTHardSigmoidMetatype,
        PTNoopMetatype,
    ]
    additional_types = ["h_sigmoid", "h_swish", "RELU"]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register("linear")
class PTLinearPruningOp(LinearPruningOp):
    subtypes = [PTLinearMetatype, PTMatMulMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register("batch_norm")
class PTBatchNormPruningOp(BatchNormPruningOp):
    subtypes = [PTBatchNormMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register("group_norm")
class PTGroupNormPruningOp(GroupNormPruningOp):
    subtypes = [PTGroupNormMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register("elementwise")
class PTElementwisePruningOp(ElementwisePruningOp):
    subtypes = [PTAddMetatype, PTSubMetatype, PTDivMetatype, PTMulMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register("stop_propagation_ops")
class PTStopMaskForwardPruningOp(StopMaskForwardPruningOp):
    subtypes = [PTMaxMetatype, PTMinMetatype, PTSumMetatype, UnknownMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register("transpose")
class PTTransposePruningOp(TransposePruningOp):
    subtypes = [PTTransposeMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register("reshape")
class PTReshape(ReshapePruningOp):
    subtypes = [PTReshapeMetatype, PTSqueezeMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register("split")
class PTSplitPruningOp(SplitPruningOp):
    subtypes = [PTSplitMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register("gather")
class PTGatherPruningOp(GatherPruningOp):
    subtypes = [PTGatherMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register("expend_as")
class PTExpandAsPruningOp(ExpandAsPruningOp):
    subtypes = [PTExpandAsMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register("masked_fill")
class PTScatterPruningOp(ScatterPruningOp):
    subtypes = [PTScatterMetatype]
