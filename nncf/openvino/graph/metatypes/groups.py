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

from nncf.openvino.graph.metatypes import openvino_metatypes as ov_metatypes

QUANTIZE_AGNOSTIC_OPERATIONS = [
    ov_metatypes.OVMaxPoolMetatype,
    ov_metatypes.OVAdaptiveMaxPoolMetatype,
    ov_metatypes.OVReduceMaxMetatype,
    ov_metatypes.OVReshapeMetatype,
    ov_metatypes.OVSqueezeMetatype,
    ov_metatypes.OVUnsqueezeMetatype,
    ov_metatypes.OVSplitMetatype,
    ov_metatypes.OVVariadicSplitMetatype,
    ov_metatypes.OVTransposeMetatype,
    ov_metatypes.OVTileMetatype,
    ov_metatypes.OVStridedSliceMetatype,
    ov_metatypes.OVShuffleChannelsMetatype,
    ov_metatypes.OVBroadcastMetatype,
    ov_metatypes.OVPadMetatype,
    ov_metatypes.OVMinimumMetatype,
    ov_metatypes.OVMaximumMetatype,
    ov_metatypes.OVConvertLikeMetatype,
    ov_metatypes.OVGatherMetatype,
    ov_metatypes.OVGatherNDMetatype,
    ov_metatypes.OVGatherElementsMetatype,
    ov_metatypes.OVScatterUpdateMetatype,
    ov_metatypes.OVScatterNDUpdateMetatype,
    ov_metatypes.OVScatterElementsUpdateMetatype,
    ov_metatypes.OVDepthToSpaceMetatype,
    ov_metatypes.OVSpaceToDepthMetatype,
    ov_metatypes.OVBatchToSpaceMetatype,
    ov_metatypes.OVSpaceToBatchMetatype,
    # ov_metatypes.OVSliceMetatype removed from the agnostic list cause of 149909 ticket.
    # OVReluMetatype is not considered to be QUANTIZATION_AGNOSTIC, because:
    # 1. Runtime doesn't provide performance benefits by quantizing the stand-alone RELU's (ticket: 59548)
    # 2. It's frequently better for the end accuracy to have quantizers set up after the RELU
    # so that the input distribution to the quantizer is non-negative
    # and we can therefore have better quantization resolution while preserving the original dynamic range
]


INPUTS_QUANTIZABLE_OPERATIONS = [
    ov_metatypes.OVConvolutionMetatype,
    ov_metatypes.OVGroupConvolutionMetatype,
    ov_metatypes.OVDepthwiseConvolutionMetatype,
    ov_metatypes.OVConvolutionBackpropDataMetatype,
    ov_metatypes.OVGroupConvolutionBackpropDataMetatype,
    ov_metatypes.OVMatMulMetatype,
    ov_metatypes.OVBatchNormMetatype,
    ov_metatypes.OVAddMetatype,
    ov_metatypes.OVSubtractMetatype,
    ov_metatypes.OVMultiplyMetatype,
    ov_metatypes.OVDivideMetatype,
    ov_metatypes.OVMaximumMetatype,
    ov_metatypes.OVMinimumMetatype,
    ov_metatypes.OVAvgPoolMetatype,
    ov_metatypes.OVAdaptiveAvgPoolMetatype,
    ov_metatypes.OVReduceMeanMetatype,
    ov_metatypes.OVMVNMetatype,
    ov_metatypes.OVNormalizeL2Metatype,
    ov_metatypes.OVInterpolateMetatype,
    ov_metatypes.OVPowerMetatype,
    ov_metatypes.OVFloorModMetatype,
    ov_metatypes.OVLessMetatype,
    ov_metatypes.OVLessEqualMetatype,
    ov_metatypes.OVGreaterMetatype,
    ov_metatypes.OVGreaterEqualMetatype,
    ov_metatypes.OVEqualMetatype,
    ov_metatypes.OVNotEqualMetatype,
    ov_metatypes.OVLogicalNotMetatype,
    ov_metatypes.OVLogicalAndMetatype,
    ov_metatypes.OVLogicalOrMetatype,
    ov_metatypes.OVLogicalXorMetatype,
    ov_metatypes.OVSquaredDifferenceMetatype,
    ov_metatypes.OVLSTMSequenceMetatype,
    ov_metatypes.OVGRUSequenceMetatype,
    ov_metatypes.OVGroupNormalizationMetatype,
    ov_metatypes.OVScaledDotProductAttentionMetatype,
]


FAKE_QUANTIZE_OPERATIONS = [ov_metatypes.OVFakeQuantizeMetatype, ov_metatypes.OVFakeConvertMetatype]


CONSTANT_OPERATIONS = [
    ov_metatypes.OVConstantMetatype,
]


SHAPEOF_OPERATIONS = [
    ov_metatypes.OVShapeOfMetatype,
]


# TODO(andrey-churkin): Can we provide a more suitable name for this variable?
LINEAR_OPERATIONS = [
    ov_metatypes.OVConvolutionMetatype,
    ov_metatypes.OVGroupConvolutionMetatype,
    ov_metatypes.OVDepthwiseConvolutionMetatype,
    ov_metatypes.OVConvolutionBackpropDataMetatype,
    ov_metatypes.OVGroupConvolutionBackpropDataMetatype,
    ov_metatypes.OVDeformableConvolutionMetatype,
    ov_metatypes.OVMatMulMetatype,
]


ATOMIC_ACTIVATIONS_OPERATIONS = [
    ov_metatypes.OVReluMetatype,
    ov_metatypes.OVClampMetatype,
    ov_metatypes.OVEluMetatype,
    ov_metatypes.OVPReluMetatype,
    ov_metatypes.OVSigmoidMetatype,
    ov_metatypes.OVHSigmoidMetatype,
    ov_metatypes.OVHardSigmoidMetatype,
    ov_metatypes.OVSwishMetatype,
    ov_metatypes.OVHSwishMetatype,
]


ARITHMETIC_OPERATIONS = [
    ov_metatypes.OVAddMetatype,
    ov_metatypes.OVSubtractMetatype,
    ov_metatypes.OVMultiplyMetatype,
    ov_metatypes.OVDivideMetatype,
]


ELEMENTWISE_OPERATIONS = [
    ov_metatypes.OVAddMetatype,
    ov_metatypes.OVMultiplyMetatype,
    ov_metatypes.OVSubtractMetatype,
    ov_metatypes.OVDivideMetatype,
    ov_metatypes.OVLessMetatype,
    ov_metatypes.OVLessEqualMetatype,
    ov_metatypes.OVGreaterMetatype,
    ov_metatypes.OVGreaterEqualMetatype,
    ov_metatypes.OVEqualMetatype,
    ov_metatypes.OVNotEqualMetatype,
    ov_metatypes.OVFloorModMetatype,
    ov_metatypes.OVLogicalOrMetatype,
    ov_metatypes.OVLogicalXorMetatype,
    ov_metatypes.OVLogicalAndMetatype,
    ov_metatypes.OVMaximumMetatype,
    ov_metatypes.OVMinimumMetatype,
]


BATCH_NORMALIZATION_OPERATIONS = [
    ov_metatypes.OVBatchNormMetatype,
]


# Keep in mind that having a metatype in this list is necessary for
# the operation to be considered as an "operation with weights", but
# it's not sufficient. For example, the MatMul operation generally has
# weights when it represents a fully connected layer. However, it can
# also be without weights. Therefore, an additional condition is needed
# to check for the existence of weights.
OPERATIONS_WITH_WEIGHTS = [
    ov_metatypes.OVConvolutionMetatype,
    ov_metatypes.OVGroupConvolutionMetatype,
    ov_metatypes.OVDepthwiseConvolutionMetatype,
    ov_metatypes.OVConvolutionBackpropDataMetatype,
    ov_metatypes.OVGroupConvolutionBackpropDataMetatype,
    ov_metatypes.OVMatMulMetatype,
    ov_metatypes.OVLSTMSequenceMetatype,
    ov_metatypes.OVGRUSequenceMetatype,
    ov_metatypes.OVEmbeddingMetatype,
]


# The same comment as mentioned above makes sense.
OPERATIONS_WITH_CONST_PORT_ID = [
    *OPERATIONS_WITH_WEIGHTS,
    ov_metatypes.OVAddMetatype,
]


# Contains the operation metatypes for which bias can be applied.
# Limited operations scope
OPERATIONS_WITH_BIAS_REDUCED = [
    ov_metatypes.OVConvolutionMetatype,
    ov_metatypes.OVMatMulMetatype,
]

OPERATIONS_WITH_BIAS = [
    *OPERATIONS_WITH_BIAS_REDUCED,
    ov_metatypes.OVDepthwiseConvolutionMetatype,
    ov_metatypes.OVConvolutionBackpropDataMetatype,
]

CONV_OPERATIONS = [
    ov_metatypes.OVConvolutionMetatype,
    ov_metatypes.OVDepthwiseConvolutionMetatype,
    ov_metatypes.OVGroupConvolutionMetatype,
    ov_metatypes.OVConvolutionBackpropDataMetatype,
    ov_metatypes.OVGroupConvolutionBackpropDataMetatype,
]

# These metatypes mix outputs for different samples into one axis.
# If reducers and aggregators collect statistics at the output of the following operations,
# assuming that 0-axis is batch axis, they get only 1 value instead of batch_size values.
# It could lead to inaccurate/incorrect statistics result.
OPERATIONS_OUTPUT_HAS_NO_BATCH_AXIS = [
    ov_metatypes.OVSpaceToBatchMetatype,
    ov_metatypes.OVBatchToSpaceMetatype,
    ov_metatypes.OVROIPoolingMetatype,
    ov_metatypes.OVROIAlignMetatype,
    ov_metatypes.OVEmbeddingMetatype,
    ov_metatypes.OVIfMetatype,
]
