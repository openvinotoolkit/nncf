from typing import List

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.common.pruning.export_helpers import (
    InputPruningOp,
    OutputPruningOp,
    IdentityMaskForwardPruningOp,
    ConvolutionPruningOp,
    TransposeConvolutionPruningOp,
    BatchNormPruningOp,
    GroupNormPruningOp,
    ConcatPruningOp,
    ElementwisePruningOp,
    StopMaskForwardPruningOp,
)


class DummyDefaultMetatype(OperatorMetatype):
    name = None

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return [cls.name]


class DummyInputMetatype(OperatorMetatype):
    name = 'input'


class DummyOutputMetatype(OperatorMetatype):
    name = 'output'


class DymmyIdentityMaskForwardMetatype(OperatorMetatype):
    name = 'identity_mask_forward'


class DummyElementwiseMetatype(OperatorMetatype):
    name = 'elementwise'


class DummyConvMetatype(OperatorMetatype):
    name = 'conv'


class DummyTransposeConvolutionMetatype(OperatorMetatype):
    name = 'transpose_conv'


class DummyBatchNormMetatype(OperatorMetatype):
    name = 'batch_norm'


class DummyGroupNormMetatype(OperatorMetatype):
    name = 'group_norm'


class DummyConcatMetatype(OperatorMetatype):
    name = 'concat'


class DummyStopPropoagtionMetatype(OperatorMetatype):
    name = 'stop_propagation_ops'


DUMMY_PRUNING_OPERATOR_METATYPES = PruningOperationsMetatypeRegistry("operator_metatypes")


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyInputMetatype.name)
class DummyInputPruningOp(InputPruningOp):
    additional_types = [DummyInputMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyOutputMetatype.name)
class DummyOutputPruningOp(OutputPruningOp):
    additional_types = [DummyOutputMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DymmyIdentityMaskForwardMetatype.name)
class DummyIdentityMaskForward(IdentityMaskForwardPruningOp):
    additional_types = [DymmyIdentityMaskForwardMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyStopPropoagtionMetatype.name)
class DummyStopMaskForward(StopMaskForwardPruningOp):
    additional_types = [DummyStopPropoagtionMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyConvMetatype.name)
class DummyConvPruningOp(ConvolutionPruningOp):
    additional_types = [DummyConvMetatype.name]


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
    StopMaskForwardOp = DummyStopMaskForward
    InputOp = DummyInputPruningOp
    additional_types = [DummyConcatMetatype.name]
