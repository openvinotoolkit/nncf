from typing import List

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.common.pruning.export_helpers import (
    OpInput,
    OpOutput,
    OpIdentityMaskForwardOps,
    OpConvolution,
    OpTransposeConvolution,
    OpBatchNorm,
    OpGroupNorm,
    OpConcat,
    OpElementwise,
    OpStopMaskForwardOps,
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
class DummyOpInput(OpInput):
    additional_types = [DummyInputMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyOutputMetatype.name)
class DummyOpOutput(OpOutput):
    additional_types = [DummyOutputMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DymmyIdentityMaskForwardMetatype.name)
class DummyOpIdentityMaskForward(OpIdentityMaskForwardOps):
    additional_types = [DymmyIdentityMaskForwardMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyStopPropoagtionMetatype.name)
class DummyOpStopMaskForward(OpStopMaskForwardOps):
    additional_types = [DummyStopPropoagtionMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyConvMetatype.name)
class DummyOpConv(OpConvolution):
    additional_types = [DummyConvMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyTransposeConvolutionMetatype.name)
class DummyOpTransposeConv(OpTransposeConvolution):
    additional_types = [DummyTransposeConvolutionMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyBatchNormMetatype.name)
class DummyOpBatchNorm(OpBatchNorm):
    additional_types = [DummyBatchNormMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyGroupNormMetatype.name)
class DummyOpGroupNorm(OpGroupNorm):
    additional_types = [DummyGroupNormMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyElementwiseMetatype.name)
class DummyOpElementwise(OpElementwise):
    additional_types = [DummyElementwiseMetatype.name]


@DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyConcatMetatype.name)
class DummyOpConcat(OpConcat):
    ConvolutionOp = DummyOpConv
    StopMaskForwardOp = DummyOpStopMaskForward
    InputOp = DummyOpInput
    additional_types = [DummyConcatMetatype.name]
