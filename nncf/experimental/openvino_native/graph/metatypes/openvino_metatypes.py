"""
 Copyright (c) 2023 Intel Corporation
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

from nncf.common.graph.operator_metatypes import INPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OUTPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatypeRegistry
from nncf.common.hardware.opset import HWConfigOpName

OV_OPERATION_METATYPES = OperatorMetatypeRegistry('openvino_operator_metatypes')


class OVOpMetatype(OperatorMetatype):
    op_names = []  # type: List[str]

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return cls.op_names


@OV_OPERATION_METATYPES.register()
class OVConvolutionMetatype(OVOpMetatype):
    name = 'ConvOp'
    op_names = ['Convolution']
    hw_config_names = [HWConfigOpName.CONVOLUTION]


@OV_OPERATION_METATYPES.register()
class OVConvolutionBackpropDataMetatype(OVOpMetatype):
    name = 'ConvTransposeOp'
    op_names = ['ConvolutionBackpropData']
    hw_config_names = [HWConfigOpName.CONVOLUTION]


@OV_OPERATION_METATYPES.register()
class OVReluMetatype(OVOpMetatype):
    name = 'ReluOp'
    op_names = ['Relu']


@OV_OPERATION_METATYPES.register()
class OVGeluMetatype(OVOpMetatype):
    name = 'GeluOp'
    op_names = ['Gelu']
    hw_config_names = [HWConfigOpName.GELU]


@OV_OPERATION_METATYPES.register()
class OVEluMetatype(OVOpMetatype):
    name = 'EluOp'
    op_names = ['Elu']


@OV_OPERATION_METATYPES.register()
class OVPReluMetatype(OVOpMetatype):
    name = 'PReluOp'
    op_names = ['PReLU']


@OV_OPERATION_METATYPES.register()
class OVSigmoidMetatype(OVOpMetatype):
    name = 'SigmoidOp'
    op_names = ['Sigmoid']


@OV_OPERATION_METATYPES.register()
class OVHardSigmoidMetatype(OVOpMetatype):
    name = 'HardSigmoidOp'
    op_names = ['HardSigmoid']


@OV_OPERATION_METATYPES.register()
class OVAveragePoolMetatype(OVOpMetatype):
    name = 'AveragePoolOp'
    op_names = ['AvgPool']
    hw_config_names = [HWConfigOpName.AVGPOOL]


@OV_OPERATION_METATYPES.register()
class OVMaxPoolMetatype(OVOpMetatype):
    name = 'MaxPoolOp'
    op_names = ['MaxPool']
    hw_config_names = [HWConfigOpName.MAXPOOL]


@OV_OPERATION_METATYPES.register()
class OVConstantMetatype(OVOpMetatype):
    name = 'ConstantOp'
    op_names = ['Constant']


@OV_OPERATION_METATYPES.register()
class OVAddMetatype(OVOpMetatype):
    name = 'AddOp'
    op_names = ['Add']
    hw_config_names = [HWConfigOpName.ADD]


@OV_OPERATION_METATYPES.register()
class OVSubMetatype(OVOpMetatype):
    name = 'SubOp'
    op_names = ['Subtract']
    hw_config_names = [HWConfigOpName.SUBTRACT]


@OV_OPERATION_METATYPES.register()
class OVMulMetatype(OVOpMetatype):
    name = 'MulOp'
    op_names = ['Multiply']
    hw_config_names = [HWConfigOpName.MULTIPLY]


@OV_OPERATION_METATYPES.register()
class OVDivMetatype(OVOpMetatype):
    name = 'DivOp'
    op_names = ['Divide']
    hw_config_names = [HWConfigOpName.DIVIDE]


@OV_OPERATION_METATYPES.register()
class OVSumMetatype(OVOpMetatype):
    name = 'SumOp'
    op_names = ['ReduceSum']
    hw_config_names = [HWConfigOpName.REDUCESUM]


@OV_OPERATION_METATYPES.register()
class OVConcatMetatype(OVOpMetatype):
    name = 'ConcatOp'
    op_names = ['Concat']
    hw_config_names = [HWConfigOpName.CONCAT]


@OV_OPERATION_METATYPES.register()
class OVBatchNormMetatype(OVOpMetatype):
    name = 'BatchNormalizationOp'
    op_names = ['BatchNormInference']


@OV_OPERATION_METATYPES.register()
class OVInterpolateMetatype(OVOpMetatype):
    name = 'InterpolateOp'
    op_names = ['Interpolate']
    hw_config_names = [HWConfigOpName.INTERPOLATE]


@OV_OPERATION_METATYPES.register()
class OVMVNMetatype(OVOpMetatype):
    name = 'MVNOp'
    op_names = ['MVN']
    hw_config_names = [HWConfigOpName.MVN]


@OV_OPERATION_METATYPES.register()
class OVNormalizeL2Metatype(OVOpMetatype):
    name = 'NormalizeL2Op'
    op_names = ['NormalizeL2']


@OV_OPERATION_METATYPES.register()
class OVReshapeMetatype(OVOpMetatype):
    name = 'ReshapeOp'
    op_names = ['Reshape']
    hw_config_names = [HWConfigOpName.RESHAPE]


@OV_OPERATION_METATYPES.register()
class OVShapeMetatype(OVOpMetatype):
    name = 'ShapeOp'
    op_names = ['ShapeOf']


@OV_OPERATION_METATYPES.register()
class OVNonZeroMetatype(OVOpMetatype):
    name = 'NonZeroOp'
    op_names = ['NonZero']


@OV_OPERATION_METATYPES.register()
class OVSplitMetatype(OVOpMetatype):
    name = 'SplitOp'
    op_names = ['Split']
    hw_config_names = [HWConfigOpName.SPLIT]


@OV_OPERATION_METATYPES.register()
class OVLessMetatype(OVOpMetatype):
    name = 'LessOp'
    op_names = ['Less']
    hw_config_names = [HWConfigOpName.LESS]


@OV_OPERATION_METATYPES.register()
class OVLessEqualMetatype(OVOpMetatype):
    name = 'LessEqualOp'
    op_names = ['LessEqual']
    hw_config_names = [HWConfigOpName.LESSEQUAL]


@OV_OPERATION_METATYPES.register()
class OVGreaterMetatype(OVOpMetatype):
    name = 'GreaterOp'
    op_names = ['Greater']
    hw_config_names = [HWConfigOpName.GREATER]


@OV_OPERATION_METATYPES.register()
class OVGreaterEqualMetatype(OVOpMetatype):
    name = 'GreaterEqualOp'
    op_names = ['GreaterEqual']
    hw_config_names = [HWConfigOpName.GREATEREQUAL]


@OV_OPERATION_METATYPES.register()
class OVEqualMetatype(OVOpMetatype):
    name = 'EqualOp'
    op_names = ['Equal']
    hw_config_names = [HWConfigOpName.EQUAL]


@OV_OPERATION_METATYPES.register()
class OVNotEqualMetatype(OVOpMetatype):
    name = 'NotEqualOp'
    op_names = ['NotEqual']
    hw_config_names = [HWConfigOpName.NOTEQUAL]


@OV_OPERATION_METATYPES.register()
class OVNotMetatype(OVOpMetatype):
    name = 'NotOp'
    op_names = ['LogicalNot']
    hw_config_names = [HWConfigOpName.LOGICALNOT]


@OV_OPERATION_METATYPES.register()
class OVAndMetatype(OVOpMetatype):
    name = 'AndOp'
    op_names = ['LogicalAnd']
    hw_config_names = [HWConfigOpName.LOGICALAND]


@OV_OPERATION_METATYPES.register()
class OVOrMetatype(OVOpMetatype):
    name = 'OrOp'
    op_names = ['LogicalOr']
    hw_config_names = [HWConfigOpName.LOGICALOR]


@OV_OPERATION_METATYPES.register()
class OVXorMetatype(OVOpMetatype):
    name = 'XorOp'
    op_names = ['LogicalXor']
    hw_config_names = [HWConfigOpName.LOGICALXOR]


@OV_OPERATION_METATYPES.register()
class OVFloorMetatype(OVOpMetatype):
    name = 'FloorOp'
    op_names = ['Floor']


@OV_OPERATION_METATYPES.register()
class OVFloorModMetatype(OVOpMetatype):
    name = 'FloorModOp'
    op_names = ['FloorMod']
    hw_config_names = [HWConfigOpName.FLOORMOD]


@OV_OPERATION_METATYPES.register()
class OVMaximumMetatype(OVOpMetatype):
    name = 'MaximumOp'
    op_names = ['Maximum']
    hw_config_names = [HWConfigOpName.MAXIMUM]

@OV_OPERATION_METATYPES.register()
class OVMinimumMetatype(OVOpMetatype):
    name = 'MinimumOp'
    op_names = ['Minimum']
    hw_config_names = [HWConfigOpName.MINIMUM]


@OV_OPERATION_METATYPES.register()
class OVSqrtMetatype(OVOpMetatype):
    name = 'SqrtOp'
    op_names = ['Sqrt']
    hw_config_names = [HWConfigOpName.POWER]


@OV_OPERATION_METATYPES.register()
class OVPowerMetatype(OVOpMetatype):
    name = 'PowerOp'
    op_names = ['Power']
    hw_config_names = [HWConfigOpName.POWER]


@OV_OPERATION_METATYPES.register()
class OVLogMetatype(OVOpMetatype):
    name = 'LogOp'
    op_names = ['Log']


@OV_OPERATION_METATYPES.register()
class OVRoiAlignMetatype(OVOpMetatype):
    name = 'RoiAlignOp'
    op_names = ['ROIAlign']


@OV_OPERATION_METATYPES.register()
class OVMatMulMetatype(OVOpMetatype):
    name = 'MatMulOp'
    op_names = ['MatMul']
    hw_config_names = [HWConfigOpName.MATMUL]


@OV_OPERATION_METATYPES.register()
class OVGatherMetatype(OVOpMetatype):
    name = 'GatherOp'
    op_names = ['Gather']


@OV_OPERATION_METATYPES.register()
class OVUnsqueezeMetatype(OVOpMetatype):
    name = 'UnsqueezeOp'
    op_names = ['Unsqueeze']
    hw_config_names = [HWConfigOpName.UNSQUEEZE]


@OV_OPERATION_METATYPES.register()
class OVSqueezeMetatype(OVOpMetatype):
    name = 'SqueezeOp'
    op_names = ['Squeeze']
    hw_config_names = [HWConfigOpName.SQUEEZE]


@OV_OPERATION_METATYPES.register()
class OVNonMaxSuppressionMetatype(OVOpMetatype):
    name = 'NonMaxSuppressionOp'
    op_names = ['NonMaxSuppression']


@OV_OPERATION_METATYPES.register()
class OVReduceMinMetatype(OVOpMetatype):
    name = 'ReduceMinOp'
    op_names = ['ReduceMin']


@OV_OPERATION_METATYPES.register()
class OVReduceMaxMetatype(OVOpMetatype):
    name = 'ReduceMaxOp'
    op_names = ['ReduceMax']
    hw_config_names = [HWConfigOpName.REDUCEMAX]


@OV_OPERATION_METATYPES.register()
class OVReduceMeanMetatype(OVOpMetatype):
    name = 'ReduceMeanOp'
    op_names = ['ReduceMean']
    hw_config_names = [HWConfigOpName.REDUCEMEAN]


@OV_OPERATION_METATYPES.register()
class OVReduceL1Metatype(OVOpMetatype):
    name = 'ReduceL1Op'
    op_names = ['ReduceL1']


@OV_OPERATION_METATYPES.register()
class OVReduceL2Metatype(OVOpMetatype):
    name = 'ReduceL2Op'
    op_names = ['ReduceL2']
    hw_config_names = [HWConfigOpName.REDUCEL2]


@OV_OPERATION_METATYPES.register()
class OVTopKMetatype(OVOpMetatype):
    name = 'TopKOp'
    op_names = ['TopK']


@OV_OPERATION_METATYPES.register()
class OVSliceMetatype(OVOpMetatype):
    name = 'SliceOp'
    op_names = ['StridedSlice']
    hw_config_names = [HWConfigOpName.STRIDEDSLICE]


@OV_OPERATION_METATYPES.register()
class OVExpMetatype(OVOpMetatype):
    name = 'ExpOp'
    op_names = ['Exp']


@OV_OPERATION_METATYPES.register()
class OVTransposeMetatype(OVOpMetatype):
    name = 'TransposeOp'
    op_names = ['Transpose']
    hw_config_names = [HWConfigOpName.TRANSPOSE]


@OV_OPERATION_METATYPES.register()
class OVTileMetatype(OVOpMetatype):
    name = 'TileOp'
    op_names = ['Tile']
    hw_config_names = [HWConfigOpName.TILE]


@OV_OPERATION_METATYPES.register()
class OVSoftmaxMetatype(OVOpMetatype):
    name = 'SoftmaxOp'
    op_names = ['SoftMax']


@OV_OPERATION_METATYPES.register()
class OVPadMetatype(OVOpMetatype):
    name = 'PadOp'
    op_names = ['Pad']
    hw_config_names = [HWConfigOpName.PAD]


@OV_OPERATION_METATYPES.register()
class OVReadValueMetatype(OVOpMetatype):
    name = 'ReadValueOp'
    op_names = ['ReadValue']


@OV_OPERATION_METATYPES.register()
class OVAssignMetatype(OVOpMetatype):
    name = 'AssignOp'
    op_names = ['Assign']


@OV_OPERATION_METATYPES.register()
@INPUT_NOOP_METATYPES.register()
class OVParameterMetatype(OVOpMetatype):
    name = 'ParameterOp'
    op_names = ['Parameter']


@OV_OPERATION_METATYPES.register()
@OUTPUT_NOOP_METATYPES.register()
class OVResultMetatype(OVOpMetatype):
    name = 'ResultOp'
    op_names = ['Result']


GENERAL_WEIGHT_LAYER_METATYPES = [OVConvolutionMetatype,
                                  OVConvolutionBackpropDataMetatype,
                                  OVMatMulMetatype]


def get_operator_metatypes() -> List[Type[OperatorMetatype]]:
    """
    Returns a list of the operator metatypes.
    :return: List of operator metatypes .
    """
    return list(OV_OPERATION_METATYPES.registry_dict.values())
