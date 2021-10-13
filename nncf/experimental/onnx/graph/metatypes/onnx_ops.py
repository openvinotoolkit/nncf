from typing import List

from nncf.common.graph.operator_metatypes import NOOP_METATYPES
from nncf.common.graph.operator_metatypes import INPUT_NOOP_METATYPES

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatypeRegistry

ONNX_OPERATION_METATYPES = OperatorMetatypeRegistry('onnx_operator_metatypes')


class ONNXOpMetatype(OperatorMetatype):
    op_names = []  # type: List[str]

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return cls.op_names


@ONNX_OPERATION_METATYPES.register()
@NOOP_METATYPES.register()
class ONNXLayerNoopMetatype(ONNXOpMetatype):
    name = 'noop'

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return [cls.name]


@ONNX_OPERATION_METATYPES.register()
@INPUT_NOOP_METATYPES.register()
class ONNXInputLayerMetatype(ONNXOpMetatype):
    name = 'InputLayer'


@ONNX_OPERATION_METATYPES.register()
class ConvolutionMetatype(ONNXOpMetatype):
    name = 'ConvOp'
    op_names = ['Conv']


@ONNX_OPERATION_METATYPES.register()
class LinearMetatype(ONNXOpMetatype):
    name = 'LinearOp'
    op_names = ['Gemm']


@ONNX_OPERATION_METATYPES.register()
class ReluMetatype(ONNXOpMetatype):
    name = 'ReluOp'
    op_names = ['Relu', 'Clip']


@ONNX_OPERATION_METATYPES.register()
class GlobalAveragePoolMetatype(ONNXOpMetatype):
    name = 'GlobalAveragePoolOp'
    op_names = ['GlobalAveragePool']


@ONNX_OPERATION_METATYPES.register()
class ConstantMetatype(ONNXOpMetatype):
    name = 'ConstantOp'
    op_names = ['Constant']


@ONNX_OPERATION_METATYPES.register()
class AddLayerMetatype(ONNXOpMetatype):
    name = 'AddOp'
    op_names = ['Add']


@ONNX_OPERATION_METATYPES.register()
class BatchNormMetatype(ONNXOpMetatype):
    name = 'BatchNormalizationOp'
    op_names = ['BatchNormalization']


GENERAL_WIGHT_LAYER_METATYPES = [ConvolutionMetatype,
                                 LinearMetatype]
