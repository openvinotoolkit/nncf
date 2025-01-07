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

from nncf.onnx.graph.metatypes import onnx_metatypes
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXOpWithWeightsMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import get_operator_metatypes

QUANTIZE_AGNOSTIC_OPERATIONS = [
    onnx_metatypes.ONNXGlobalMaxPoolMetatype,
    onnx_metatypes.ONNXMaxPoolMetatype,
    onnx_metatypes.ONNXReduceMaxMetatype,
    onnx_metatypes.ONNXReshapeMetatype,
    onnx_metatypes.ONNXTransposeMetatype,
    onnx_metatypes.ONNXSqueezeMetatype,
    onnx_metatypes.ONNXUnsqueezeMetatype,
    onnx_metatypes.ONNXSplitMetatype,
    onnx_metatypes.ONNXTileMetatype,
    onnx_metatypes.ONNXCenterCropPadMetatype,
    onnx_metatypes.ONNXSliceMetatype,
    onnx_metatypes.ONNXPadMetatype,
    onnx_metatypes.ONNXGatherMetatype,
    onnx_metatypes.ONNXGatherNDMetatype,
    onnx_metatypes.ONNXGatherElementsMetatype,
    onnx_metatypes.ONNXDepthToSpaceMetatype,
    onnx_metatypes.ONNXSpaceToDepthMetatype,
    onnx_metatypes.ONNXScatterElementsMetatype,
    onnx_metatypes.ONNXScatterNDMetatype,
    onnx_metatypes.ONNXScatterMetatype,
    onnx_metatypes.ONNXCastLikeMetatype,
    onnx_metatypes.ONNXDropoutMetatype,
    onnx_metatypes.ONNXFlattenMetatype,
    onnx_metatypes.ONNXExpandMetatype,
    onnx_metatypes.ONNXIdentityMetatype,
    # ONNXReluMetatype is not considered to be QUANTIZATION_AGNOSTIC, because:
    # 1. Runtime doesn't provide performance benefits by quantizing the stand-alone RELU's (ticket: 59548)
    # 2. It's frequently better for the end accuracy to have quantizers set up after the RELU
    # so that the input distribution to the quantizer is non-negative
    # and we can therefore have better quantization resolution while preserving the original dynamic range
]


MATMUL_METATYPES = [onnx_metatypes.ONNXGemmMetatype, onnx_metatypes.ONNXMatMulMetatype]


INPUTS_QUANTIZABLE_OPERATIONS = [
    onnx_metatypes.ONNXConvolutionMetatype,
    onnx_metatypes.ONNXDepthwiseConvolutionMetatype,
    onnx_metatypes.ONNXConvolutionTransposeMetatype,
    *MATMUL_METATYPES,
    onnx_metatypes.ONNXAveragePoolMetatype,
    onnx_metatypes.ONNXGlobalAveragePoolMetatype,
    onnx_metatypes.ONNXAddLayerMetatype,
    onnx_metatypes.ONNXSubMetatype,
    onnx_metatypes.ONNXMulLayerMetatype,
    onnx_metatypes.ONNXBatchNormMetatype,
    onnx_metatypes.ONNXHardSigmoidMetatype,
    onnx_metatypes.ONNXResizeMetatype,
    onnx_metatypes.ONNXPowMetatype,
    onnx_metatypes.ONNXReciprocalMetatype,
    onnx_metatypes.ONNXMaximumMetatype,
    onnx_metatypes.ONNXMinimumMetatype,
]

CONSTANT_WEIGHT_LAYER_METATYPES = [
    metatype
    for metatype in get_operator_metatypes()
    if issubclass(metatype, ONNXOpWithWeightsMetatype) and metatype.weight_port_ids
]

POSSIBLE_WEIGHT_LAYER_METATYPES = [
    metatype
    for metatype in get_operator_metatypes()
    if issubclass(metatype, ONNXOpWithWeightsMetatype) and metatype.possible_weight_ports
]

OPERATIONS_WITH_WEIGHTS = list(set().union(CONSTANT_WEIGHT_LAYER_METATYPES, POSSIBLE_WEIGHT_LAYER_METATYPES))

LINEAR_OPERATIONS = [
    onnx_metatypes.ONNXConvolutionMetatype,
    onnx_metatypes.ONNXDepthwiseConvolutionMetatype,
    onnx_metatypes.ONNXConvolutionTransposeMetatype,
    onnx_metatypes.ONNXDeformableConvolutionMetatype,
    *MATMUL_METATYPES,
]


ATOMIC_ACTIVATIONS_OPERATIONS = [
    onnx_metatypes.ONNXReluMetatype,
    onnx_metatypes.ONNXLeakyReluMetatype,
    onnx_metatypes.ONNXThresholdedReluMetatype,
    onnx_metatypes.ONNXEluMetatype,
    onnx_metatypes.ONNXPReluMetatype,
    onnx_metatypes.ONNXSigmoidMetatype,
    onnx_metatypes.ONNXHardSigmoidMetatype,
    onnx_metatypes.ONNXHardSwishMetatype,
]


ARITHMETIC_OPERATIONS = [
    onnx_metatypes.ONNXAddLayerMetatype,
    onnx_metatypes.ONNXSubMetatype,
    onnx_metatypes.ONNXMulLayerMetatype,
    onnx_metatypes.ONNXDivLayerMetatype,
]

ELEMENTWISE_OPERATIONS = [
    onnx_metatypes.ONNXAddLayerMetatype,
    onnx_metatypes.ONNXMulLayerMetatype,
    onnx_metatypes.ONNXSubMetatype,
    onnx_metatypes.ONNXDivLayerMetatype,
    onnx_metatypes.ONNXLessMetatype,
    onnx_metatypes.ONNXLessOrEqualMetatype,
    onnx_metatypes.ONNXGreaterMetatype,
    onnx_metatypes.ONNXGreaterOrEqualMetatype,
    onnx_metatypes.ONNXEqualMetatype,
    onnx_metatypes.ONNXModMetatype,
    onnx_metatypes.ONNXOrMetatype,
    onnx_metatypes.ONNXNotMetatype,
    onnx_metatypes.ONNXAndMetatype,
    onnx_metatypes.ONNXXOrMetatype,
    onnx_metatypes.ONNXMaximumMetatype,
    onnx_metatypes.ONNXMinimumMetatype,
    onnx_metatypes.ONNXMeanMetatype,
]


BATCH_NORMALIZATION_OPERATIONS = [
    onnx_metatypes.ONNXBatchNormMetatype,
]


# Contains the operation metatypes for which bias can be applied.
OPERATIONS_WITH_BIAS_REDUCED = [
    onnx_metatypes.ONNXConvolutionMetatype,
    onnx_metatypes.ONNXGemmMetatype,
    # TODO: Need to add MatMul with the separate bias support (CVS-135433)
]

OPERATIONS_WITH_BIAS = [
    *OPERATIONS_WITH_BIAS_REDUCED,
    onnx_metatypes.ONNXDepthwiseConvolutionMetatype,
    onnx_metatypes.ONNXConvolutionTransposeMetatype,
]


QUANTIZE_DEQUANTIZE_OPERATIONS = [
    onnx_metatypes.ONNXQuantizeLinearMetatype,
    onnx_metatypes.ONNXDequantizeLinearMetatype,
]

# These metatypes mix outputs for different samples into one axis.
# If reducers and aggregators collect statistics at the output of the following operations,
# assuming that 0-axis is batch axis, they get only 1 value instead of batch_size values.
# It could lead to inaccurate/incorrect statistics result.
OPERATIONS_OUTPUT_HAS_NO_BATCH_AXIS = [
    onnx_metatypes.ONNXROIAlignMetatype,
    onnx_metatypes.ONNXEmbeddingMetatype,
]
