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


from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.torch.graph.operator_metatypes import (
    PTAddMetatype,
    PTAvgPool2dMetatype,
    PTBatchNormMetatype,
    PTDivMetatype,
    PTDropoutMetatype,
    PTELUMetatype,
    PTExpandAsMetatype,
    PTRELU6Metatype,
    PTGELUMetatype,
    PTGroupNormMetatype,
    PTHardTanhMetatype,
    PTHardSwishMetatype,
    PTHardSigmoidMetatype,
    PTInputNoopMetatype,
    PTLinearMetatype,
    PTMatMulMetatype,
    PTMaxMetatype,
    PTMaxPool2dMetatype,
    PTMeanMetatype,
    PTMinMetatype,
    PTMulMetatype,
    PTNoopMetatype,
    PTOutputNoopMetatype,
    PTPowerMetatype,
    PTPRELUMetatype,
    PTLeakyRELUMetatype,
    PTRELUMetatype,
    PTScatterMetatype,
    PTSigmoidMetatype,
    PTSILUMetatype,
    PTSoftmaxMetatype,
    PTSubMetatype,
    PTSumMetatype,
    PTTanhMetatype,
    PTReshapeMetatype,
    PTTransposeMetatype,
    PTSplitMetatype,
    PTGatherMetatype
)
from nncf.experimental.common.pruning.operations import (
    ExpandAsPruningOp,
    InputPruningOp,
    OutputPruningOp,
    IdentityMaskForwardPruningOp,
    ConvolutionPruningOp,
    ScatterPruningOp,
    TransposeConvolutionPruningOp,
    BatchNormPruningOp,
    LinearPruningOp,
    GroupNormPruningOp,
    ElementwisePruningOp,
    ReshapePruningOp,
    TransposePruningOp,
    StopMaskForwardPruningOp,
    SplitPruningOp,
    GatherPruningOp
)

from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry

PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES = PruningOperationsMetatypeRegistry("operator_metatypes")


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register('model_input')
class PTInputPruningOp(InputPruningOp):
    subtypes = [PTInputNoopMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register('model_output')
class PTOutputPruningOp(OutputPruningOp):
    subtypes = [PTOutputNoopMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register('identity_mask_propagation')
class PTIdentityMaskForwardPruningOp(IdentityMaskForwardPruningOp):
    subtypes = [PTHardTanhMetatype, PTTanhMetatype, PTRELUMetatype, PTRELU6Metatype, PTLeakyRELUMetatype,
                PTPRELUMetatype, PTELUMetatype, PTGELUMetatype, PTSigmoidMetatype, PTSoftmaxMetatype,
                PTAvgPool2dMetatype, PTMaxPool2dMetatype, PTDropoutMetatype, PTSILUMetatype, PTPowerMetatype,
                PTHardSwishMetatype, PTHardSigmoidMetatype, PTNoopMetatype]
    additional_types = ['h_sigmoid', 'h_swish', 'RELU']


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register('linear')
class PTLinearPruningOp(LinearPruningOp):
    subtypes = [PTLinearMetatype, PTMatMulMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register('batch_norm')
class PTBatchNormPruningOp(BatchNormPruningOp):
    subtypes = [PTBatchNormMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register('group_norm')
class PTGroupNormPruningOp(GroupNormPruningOp):
    subtypes = [PTGroupNormMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register('elementwise')
class PTElementwisePruningOp(ElementwisePruningOp):
    subtypes = [PTAddMetatype, PTSubMetatype, PTDivMetatype, PTMulMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register('stop_propagation_ops')
class PTStopMaskForwardPruningOp(StopMaskForwardPruningOp):
    subtypes = [PTMeanMetatype, PTMaxMetatype, PTMinMetatype, PTSumMetatype,
                UnknownMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register('transpose')
class PTTransposePruningOp(TransposePruningOp):
    subtypes = [PTTransposeMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register('reshape')
class PTReshape(ReshapePruningOp):
    subtypes = [PTReshapeMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register('split')
class PTSplitPruningOp(SplitPruningOp):
    subtypes = [PTSplitMetatype]


@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register('gather')
class PTGatherPruningOp(GatherPruningOp):
    subtypes = [PTGatherMetatype]

@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register('expend_as')
class PTExpandAsPruningOp(ExpandAsPruningOp):
    subtypes = [PTExpandAsMetatype]

@PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES.register('masked_fill')
class PTScatterPruningOp(ScatterPruningOp):
    subtypes = [PTScatterMetatype]

