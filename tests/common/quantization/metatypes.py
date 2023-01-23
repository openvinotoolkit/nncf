from typing import List

from nncf.common.graph import OperatorMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatypeRegistry
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait

METATYPES_FOR_TEST = OperatorMetatypeRegistry('TEST_METATYPES')


class TestMetatype(OperatorMetatype):
    name = None

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        assert cls.name is not None
        return [cls.name]


@METATYPES_FOR_TEST.register()
class BatchNormTestMetatype(TestMetatype):
    name = 'batch_norm'


@METATYPES_FOR_TEST.register()
class Conv2dTestMetatype(TestMetatype):
    name = 'conv2d'


@METATYPES_FOR_TEST.register()
class MatMulTestMetatype(TestMetatype):
    name = 'matmul'


@METATYPES_FOR_TEST.register()
class MaxPool2dTestMetatype(TestMetatype):
    name = 'max_pool2d'


@METATYPES_FOR_TEST.register()
class GeluTestMetatype(TestMetatype):
    name = 'gelu'


@METATYPES_FOR_TEST.register()
class DropoutTestMetatype(TestMetatype):
    name = 'dropout'


@METATYPES_FOR_TEST.register()
class MinTestMetatype(TestMetatype):
    name = 'min'


@METATYPES_FOR_TEST.register()
class SoftmaxTestMetatype(TestMetatype):
    name = 'softmax'


@METATYPES_FOR_TEST.register()
class CatTestMetatype(TestMetatype):
    name = 'cat'


@METATYPES_FOR_TEST.register()
class LinearTestMetatype(TestMetatype):
    name = 'linear'


@METATYPES_FOR_TEST.register()
class TopKTestMetatype(TestMetatype):
    name = 'topk'


@METATYPES_FOR_TEST.register()
class NMSTestMetatype(TestMetatype):
    name = 'nms'


@METATYPES_FOR_TEST.register()
class IdentityTestMetatype(TestMetatype):
    name = 'identity'


@METATYPES_FOR_TEST.register()
class ReshapeTestMetatype(TestMetatype):
    name = 'reshape'


@METATYPES_FOR_TEST.register()
class QuantizerTestMetatype(TestMetatype):
    name = 'quantizer'


@METATYPES_FOR_TEST.register()
class ConstantTestMetatype(TestMetatype):
    name = 'constant'


@METATYPES_FOR_TEST.register()
class ReluTestMetatype(TestMetatype):
    name = 'relu'


@METATYPES_FOR_TEST.register()
class AddTestMetatype(TestMetatype):
    name = 'add'


WEIGHT_LAYER_METATYPES = [LinearTestMetatype, Conv2dTestMetatype, MatMulTestMetatype]

DEFAULT_TEST_QUANT_TRAIT_MAP = {
    QuantizationTrait.INPUTS_QUANTIZABLE: [
        BatchNormTestMetatype,
        Conv2dTestMetatype,
        MatMulTestMetatype,
        GeluTestMetatype,
        LinearTestMetatype,
        AddTestMetatype,
    ],
    QuantizationTrait.NON_QUANTIZABLE: [
        MaxPool2dTestMetatype,
        DropoutTestMetatype,
        MinTestMetatype,
        SoftmaxTestMetatype,
        UnknownMetatype
    ],
    QuantizationTrait.CONCAT: [
        CatTestMetatype
    ],
}


QUANTIZER_METATYPES = [
    QuantizerTestMetatype,
]


CONSTANT_METATYPES = [
    ConstantTestMetatype,
]


QUANTIZABLE_METATYPES = [
    Conv2dTestMetatype,
    AddTestMetatype,
]


QUANTIZE_AGNOSTIC_METATYPES = [
    MaxPool2dTestMetatype,
    ReluTestMetatype,
]
