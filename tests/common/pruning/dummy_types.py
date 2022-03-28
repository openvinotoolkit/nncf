from typing import List

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.common.pruning.operations import (
    InputPruningOp,
    OutputPruningOp,
    IdentityMaskForwardPruningOp,
    ConvolutionPruningOp,
    LinearPruningOp,
    TransposeConvolutionPruningOp,
    BatchNormPruningOp,
    GroupNormPruningOp,
    ConcatPruningOp,
    ElementwisePruningOp,
    ReshapePruningOp,
    FlattenPruningOp,
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


class DummyIdentityMaskForwardMetatype(OperatorMetatype):
    name = 'identity_mask_forward'


class DummyElementwiseMetatype(OperatorMetatype):
    name = 'elementwise'


class DummyConvMetatype(OperatorMetatype):
    name = 'conv'

class DummyLinearMetatype(OperatorMetatype):
    name = 'linear'

class DummyTransposeConvolutionMetatype(OperatorMetatype):
    name = 'transpose_conv'


class DummyBatchNormMetatype(OperatorMetatype):
    name = 'batch_norm'


class DummyGroupNormMetatype(OperatorMetatype):
    name = 'group_norm'


class DummyConcatMetatype(OperatorMetatype):
    name = 'concat'


class DummyStopMaskForwardMetatype(OperatorMetatype):
    name = 'stop_propagation_ops'


class DummyReshapeMetatye(OperatorMetatype):
    name = 'reshape'


class DummyFlattenMetatype(OperatorMetatype):
    name = 'flatten'


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
