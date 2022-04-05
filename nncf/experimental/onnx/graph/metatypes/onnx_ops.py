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
class ONNXConvolutionMetatype(ONNXOpMetatype):
    name = 'ConvOp'
    op_names = ['Conv']
    hw_config_names = [HWConfigOpName.CONVOLUTION]


@ONNX_OPERATION_METATYPES.register()
class ONNXLinearMetatype(ONNXOpMetatype):
    name = 'LinearOp'
    op_names = ['Gemm']
    hw_config_names = [HWConfigOpName.MATMUL]


@ONNX_OPERATION_METATYPES.register()
class ONNXReluMetatype(ONNXOpMetatype):
    name = 'ReluOp'
    op_names = ['Relu', 'Clip']


@ONNX_OPERATION_METATYPES.register()
class ONNXSigmoidMetatype(ONNXOpMetatype):
    name = 'SigmoidOp'
    op_names = ['Sigmoid']


@ONNX_OPERATION_METATYPES.register()
class ONNXHardSigmoidMetatype(ONNXOpMetatype):
    name = 'HardSigmoidOp'
    op_names = ['HardSigmoid']


@ONNX_OPERATION_METATYPES.register()
class ONNXGlobalAveragePoolMetatype(ONNXOpMetatype):
    name = 'GlobalAveragePoolOp'
    op_names = ['GlobalAveragePool']
    hw_config_names = [HWConfigOpName.AVGPOOL]


@ONNX_OPERATION_METATYPES.register()
class ONNXAveragePoolMetatype(ONNXOpMetatype):
    name = 'AveragePoolOp'
    op_names = ['AveragePool']
    hw_config_names = [HWConfigOpName.AVGPOOL]


@ONNX_OPERATION_METATYPES.register()
class ONNXMaxPoolMetatype(ONNXOpMetatype):
    name = 'MaxPoolOp'
    op_names = ['MaxPool']
    hw_config_names = [HWConfigOpName.MAXPOOL]


@ONNX_OPERATION_METATYPES.register()
class ONNXConstantMetatype(ONNXOpMetatype):
    name = 'ConstantOp'
    op_names = ['Constant']


@ONNX_OPERATION_METATYPES.register()
class ONNXAddLayerMetatype(ONNXOpMetatype):
    name = 'AddOp'
    op_names = ['Add']
    hw_config_names = [HWConfigOpName.ADD]


@ONNX_OPERATION_METATYPES.register()
class ONNXMulLayerMetatype(ONNXOpMetatype):
    name = 'MulOp'
    op_names = ['Mul']
    hw_config_names = [HWConfigOpName.MULTIPLY]


@ONNX_OPERATION_METATYPES.register()
class ONNXSumMetatype(ONNXOpMetatype):
    name = 'SumOp'
    op_names = ['Sum']
    hw_config_names = [HWConfigOpName.REDUCESUM]


@ONNX_OPERATION_METATYPES.register()
class ONNXConcatLayerMetatype(ONNXOpMetatype):
    name = 'ConcatOp'
    op_names = ['Concat']
    hw_config_names = [HWConfigOpName.CONCAT]


@ONNX_OPERATION_METATYPES.register()
class ONNXBatchNormMetatype(ONNXOpMetatype):
    name = 'BatchNormalizationOp'
    op_names = ['BatchNormalization']


@ONNX_OPERATION_METATYPES.register()
class ONNXResizeMetatype(ONNXOpMetatype):
    name = 'ResizeOp'
    op_names = ['Resize']
    hw_config_names = [HWConfigOpName.INTERPOLATE]


@ONNX_OPERATION_METATYPES.register()
class ONNXReshapeMetatype(ONNXOpMetatype):
    name = 'ReshapeOp'
    op_names = ['Reshape']
    hw_config_names = [HWConfigOpName.RESHAPE]


@ONNX_OPERATION_METATYPES.register()
class ONNXTransposeMetatype(ONNXOpMetatype):
    name = 'TransposeOp'
    op_names = ['Transpose']
    hw_config_names = [HWConfigOpName.TRANSPOSE]


@ONNX_OPERATION_METATYPES.register()
class ONNXFlattenMetatype(ONNXOpMetatype):
    name = 'FlattenOp'
    op_names = ['Flatten']
    hw_config_names = [HWConfigOpName.FLATTEN]


@ONNX_OPERATION_METATYPES.register()
class ONNXSoftmaxMetatype(ONNXOpMetatype):
    name = 'SoftmaxOp'
    op_names = ['Softmax']


GENERAL_WEIGHT_LAYER_METATYPES = [ONNXConvolutionMetatype,
                                  ONNXLinearMetatype]


def get_operator_metatypes() -> List[Type[OperatorMetatype]]:
    """
    Returns a list of the operator metatypes.

    :return: List of operator metatypes .
    """
    return list(ONNX_OPERATION_METATYPES.registry_dict.values())
