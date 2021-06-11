"""
 Copyright (c) 2021 Intel Corporation
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

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatypeRegistry
from nncf.common.hardware.opset import HWConfigOpName

TF_OPERATION_METATYPES = OperatorMetatypeRegistry('tf_operation_metatypes')


class TFOpMetatype(OperatorMetatype):
    op_names = []  # type: List[str]

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return cls.op_names


@TF_OPERATION_METATYPES.register()
class NoopMetatype(OperatorMetatype):
    name = 'noop'

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return [cls.name]


@TF_OPERATION_METATYPES.register()
class TFIdentityOpMetatype(TFOpMetatype):
    name = 'IdentityOp'
    op_names = ['Identity']


@TF_OPERATION_METATYPES.register()
class TFPackOpMetatype(TFOpMetatype):
    # Unsqueezes->Concat pattern
    name = 'PackOp'
    op_names = ['Pack']


@TF_OPERATION_METATYPES.register()
class TFPadOpMetatype(TFOpMetatype):
    name = 'PadOp'
    op_names = ['Pad']
    hw_config_names = [HWConfigOpName.PAD]


@TF_OPERATION_METATYPES.register()
class TFStridedSliceOpMetatype(TFOpMetatype):
    name = 'StridedSliceOp'
    op_names = ['StridedSlice']
    hw_config_names = [HWConfigOpName.STRIDEDSLICE]


@TF_OPERATION_METATYPES.register()
class TFConcatOpMetatype(TFOpMetatype):
    name = 'ConcatOp'
    op_names = ['Concat', 'ConcatV2']
    hw_config_names = [HWConfigOpName.CONCAT]


@TF_OPERATION_METATYPES.register()
class TFAddOpMetatype(TFOpMetatype):
    name = 'AddOp'
    op_names = ['Add', 'AddV2']
    hw_config_names = [HWConfigOpName.ADD]


@TF_OPERATION_METATYPES.register()
class TFSubOpMetatype(TFOpMetatype):
    name = 'SubOp'
    op_names = ['Sub']
    hw_config_names = [HWConfigOpName.SUBTRACT]


@TF_OPERATION_METATYPES.register()
class TFMulOpMetatype(TFOpMetatype):
    name = 'MulOp'
    op_names = ['Mul']
    hw_config_names = [HWConfigOpName.MULTIPLY]


@TF_OPERATION_METATYPES.register()
class TFAvgPoolOpMetatype(TFOpMetatype):
    name = 'AvgPoolOp'
    op_names = ['AvgPool']
    hw_config_names = [HWConfigOpName.AVGPOOL]


@TF_OPERATION_METATYPES.register()
class TFAvgPool3DOpMetatype(TFOpMetatype):
    name = 'AvgPool3DOp'
    op_names = ['AvgPool3D']
    hw_config_names = [HWConfigOpName.AVGPOOL]


@TF_OPERATION_METATYPES.register()
class TFReluOpMetatype(TFOpMetatype):
    name = 'ReluOp'
    op_names = ['Relu']


@TF_OPERATION_METATYPES.register()
class TFRelu6OpMetatype(TFOpMetatype):
    name = 'Relu6Op'
    op_names = ['Relu6']


@TF_OPERATION_METATYPES.register()
class TFMatMulOpMetatype(TFOpMetatype):
    name = 'MatMulOp'
    op_names = ['MatMul']


@TF_OPERATION_METATYPES.register()
class TFConv2DOpMetatype(TFOpMetatype):
    name = 'Conv2DOp'
    op_names = ['Conv2D']


@TF_OPERATION_METATYPES.register()
class TFConv3DOpMetatype(TFOpMetatype):
    name = 'Conv3DOp'
    op_names = ['Conv3D']


@TF_OPERATION_METATYPES.register()
class TFDepthwiseConv2dNativeOpMetatype(TFOpMetatype):
    name = 'DepthwiseConv2dNativeOp'
    op_names = ['DepthwiseConv2dNative']


@TF_OPERATION_METATYPES.register()
class TFQuantizedConv2DOpMetatype(TFOpMetatype):
    name = 'QuantizedConv2DOp'
    op_names = ['QuantizedConv2D']


@TF_OPERATION_METATYPES.register()
class TFReshapeOpMetatype(TFOpMetatype):
    name = 'ReshapeOp'
    op_names = ['Reshape']
    hw_config_names = [HWConfigOpName.RESHAPE]
