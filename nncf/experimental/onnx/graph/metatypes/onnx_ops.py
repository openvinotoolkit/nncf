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

from typing import List
from typing import Type

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatypeRegistry
from nncf.common.hardware.opset import HWConfigOpName

ONNX_OPERATION_METATYPES = OperatorMetatypeRegistry('onnx_operator_metatypes')


class ONNXOpMetatype(OperatorMetatype):
    op_names = []  # type: List[str]

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return cls.op_names


@ONNX_OPERATION_METATYPES.register()
class ConvolutionMetatype(ONNXOpMetatype):
    name = 'ConvOp'
    op_names = ['Conv']
    hw_config_names = [HWConfigOpName.CONVOLUTION]


@ONNX_OPERATION_METATYPES.register()
class LinearMetatype(ONNXOpMetatype):
    name = 'LinearOp'
    op_names = ['Gemm']
    hw_config_names = [HWConfigOpName.MATMUL]


@ONNX_OPERATION_METATYPES.register()
class ReluMetatype(ONNXOpMetatype):
    name = 'ReluOp'
    op_names = ['Relu', 'Clip']


@ONNX_OPERATION_METATYPES.register()
class SigmoidMetatype(ONNXOpMetatype):
    name = 'SigmoidOp'
    op_names = ['Sigmoid']


@ONNX_OPERATION_METATYPES.register()
class GlobalAveragePoolMetatype(ONNXOpMetatype):
    name = 'GlobalAveragePoolOp'
    op_names = ['GlobalAveragePool']
    hw_config_names = [HWConfigOpName.AVGPOOL]


@ONNX_OPERATION_METATYPES.register()
class MaxPoolMetatype(ONNXOpMetatype):
    name = 'MaxPoolOp'
    op_names = ['MaxPool']
    hw_config_names = [HWConfigOpName.MAXPOOL]


@ONNX_OPERATION_METATYPES.register()
class ConstantMetatype(ONNXOpMetatype):
    name = 'ConstantOp'
    op_names = ['Constant']


@ONNX_OPERATION_METATYPES.register()
class AddLayerMetatype(ONNXOpMetatype):
    name = 'AddOp'
    op_names = ['Add']
    hw_config_names = [HWConfigOpName.ADD]


@ONNX_OPERATION_METATYPES.register()
class MulLayerMetatype(ONNXOpMetatype):
    name = 'MulOp'
    op_names = ['Mul']
    hw_config_names = [HWConfigOpName.MULTIPLY]


@ONNX_OPERATION_METATYPES.register()
class SumMetatype(ONNXOpMetatype):
    name = 'SumOp'
    op_names = ['Sum']
    hw_config_names = [HWConfigOpName.REDUCESUM]


@ONNX_OPERATION_METATYPES.register()
class ConcatLayerMetatype(ONNXOpMetatype):
    name = 'ConcatOp'
    op_names = ['Concat']
    hw_config_names = [HWConfigOpName.CONCAT]


@ONNX_OPERATION_METATYPES.register()
class BatchNormMetatype(ONNXOpMetatype):
    name = 'BatchNormalizationOp'
    op_names = ['BatchNormalization']


@ONNX_OPERATION_METATYPES.register()
class ResizeMetatype(ONNXOpMetatype):
    name = 'ResizeOp'
    op_names = ['Resize']


@ONNX_OPERATION_METATYPES.register()
class ReshapeMetatype(ONNXOpMetatype):
    name = 'ReshapeOp'
    op_names = ['Reshape']
    hw_config_names = [HWConfigOpName.RESHAPE]


@ONNX_OPERATION_METATYPES.register()
class TransposeMetatype(ONNXOpMetatype):
    name = 'TransposeOp'
    op_names = ['Transpose']
    hw_config_names = [HWConfigOpName.TRANSPOSE]


@ONNX_OPERATION_METATYPES.register()
class FlattenMetatype(ONNXOpMetatype):
    name = 'FlattenOp'
    op_names = ['Flatten']
    hw_config_names = [HWConfigOpName.FLATTEN]


GENERAL_WEIGHT_LAYER_METATYPES = [ConvolutionMetatype,
                                  LinearMetatype]


def get_operator_metatypes() -> List[Type[OperatorMetatype]]:
    """
    Returns a list of the operator metatypes.

    :return: List of operator metatypes .
    """
    return list(ONNX_OPERATION_METATYPES.registry_dict.values())
