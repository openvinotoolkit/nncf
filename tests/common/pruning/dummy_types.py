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

from typing import List

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.pruning.operations import BatchNormPruningOp
from nncf.common.pruning.operations import ConcatPruningOp
from nncf.common.pruning.operations import ConvolutionPruningOp
from nncf.common.pruning.operations import ElementwisePruningOp
from nncf.common.pruning.operations import FlattenPruningOp
from nncf.common.pruning.operations import GroupNormPruningOp
from nncf.common.pruning.operations import IdentityMaskForwardPruningOp
from nncf.common.pruning.operations import InputPruningOp
from nncf.common.pruning.operations import LinearPruningOp
from nncf.common.pruning.operations import OutputPruningOp
from nncf.common.pruning.operations import ReshapePruningOp
from nncf.common.pruning.operations import SplitPruningOp
from nncf.common.pruning.operations import StopMaskForwardPruningOp
from nncf.common.pruning.operations import TransposeConvolutionPruningOp
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry


class DummyDefaultMetatype(OperatorMetatype):
    name = None

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return [cls.name]


class DummyInputMetatype(OperatorMetatype):
    name = "input"


class DummyOutputMetatype(OperatorMetatype):
    name = "output"


class DummyIdentityMaskForwardMetatype(OperatorMetatype):
    name = "identity_mask_forward"


class DummyElementwiseMetatype(OperatorMetatype):
    name = "elementwise"


class DummyConvMetatype(OperatorMetatype):
    name = "conv"


class DummyLinearMetatype(OperatorMetatype):
    name = "linear"


class DummyTransposeConvolutionMetatype(OperatorMetatype):
    name = "transpose_conv"


class DummyBatchNormMetatype(OperatorMetatype):
    name = "batch_norm"


class DummyGroupNormMetatype(OperatorMetatype):
    name = "group_norm"


class DummyConcatMetatype(OperatorMetatype):
    name = "concat"


class DummyStopMaskForwardMetatype(OperatorMetatype):
    name = "stop_propagation_ops"


class DummyReshapeMetatye(OperatorMetatype):
    name = "reshape"


class DummyFlattenMetatype(OperatorMetatype):
    name = "flatten"


class DummySplitMetatype(OperatorMetatype):
    name = "chunk"


DUMMY_PRUNING_OPERATOR_METATYPES = PruningOperationsMetatypeRegistry("operator_metatypes")


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyInputMetatype.name)
class DummyInputPruningOp(InputPruningOp):
    additional_types = [DummyInputMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyOutputMetatype.name)
class DummyOutputPruningOp(OutputPruningOp):
    additional_types = [DummyOutputMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyIdentityMaskForwardMetatype.name)
class DummyIdentityMaskForward(IdentityMaskForwardPruningOp):
    additional_types = [DummyIdentityMaskForwardMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyStopMaskForwardMetatype.name)
class DummyStopMaskForwardPruningOp(StopMaskForwardPruningOp):
    additional_types = [DummyStopMaskForwardMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyConvMetatype.name)
class DummyConvPruningOp(ConvolutionPruningOp):
    additional_types = [DummyConvMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyLinearMetatype.name)
class DummyLinearPruningOp(LinearPruningOp):
    additional_types = [DummyLinearMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyTransposeConvolutionMetatype.name)
class DummyTransposeConvPruningOp(TransposeConvolutionPruningOp):
    additional_types = [DummyTransposeConvolutionMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyBatchNormMetatype.name)
class DummyBatchNormPruningOp(BatchNormPruningOp):
    additional_types = [DummyBatchNormMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyGroupNormMetatype.name)
class DummyGroupNormPruningOp(GroupNormPruningOp):
    additional_types = [DummyGroupNormMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyElementwiseMetatype.name)
class DummyElementwisePruningOp(ElementwisePruningOp):
    additional_types = [DummyElementwiseMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyConcatMetatype.name)
class DummyConcatPruningOp(ConcatPruningOp):
    ConvolutionOp = DummyConvPruningOp
    StopMaskForwardOp = DummyStopMaskForwardPruningOp
    InputOp = DummyInputPruningOp
    additional_types = [DummyConcatMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyReshapeMetatye.name)
class DummyReshapePruningOp(ReshapePruningOp):
    additional_types = [DummyReshapeMetatye.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyFlattenMetatype.name)
class DummyFlattenPruningOp(FlattenPruningOp):
    additional_types = [DummyFlattenMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummySplitMetatype.name)
class DummySplitPruningOp(SplitPruningOp):
    additional_types = [DummySplitMetatype.name]
