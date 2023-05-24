# Copyright (c) 2023 Intel Corporation
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
    # OVReluMetatype is not considered to be QUANTIZATION_AGNOSTIC, because:
    # 1. Runtime doesn't provide performance benefits by quantizing the stand-alone RELU's (ticket: 59548)
    # 2. It's frequently better for the end accuracy to have quantizers set up after the RELU
    # so that the input distribution to the quantizer is non-negative
    # and we can therefore have better quantization resolution while preserving the original dynamic range
]


QUANTIZABLE_OPERATIONS = [
    ov_metatypes.OVConvolutionMetatype,
    ov_metatypes.OVGroupConvolutionMetatype,
    ov_metatypes.OVDepthwiseConvolutionMetatype,
    ov_metatypes.OVConvolutionBackpropDataMetatype,
    ov_metatypes.OVGroupConvolutionBackpropDataMetatype,
    ov_metatypes.OVMatMulMetatype,
    ov_metatypes.OVAddMetatype,
    ov_metatypes.OVMultiplyMetatype,
    ov_metatypes.OVLessMetatype,
    ov_metatypes.OVLessEqualMetatype,
    ov_metatypes.OVGreaterMetatype,
    ov_metatypes.OVGreaterEqualMetatype,
    ov_metatypes.OVDivideMetatype,
    ov_metatypes.OVEqualMetatype,
    ov_metatypes.OVSubtractMetatype,
    ov_metatypes.OVNotEqualMetatype,
    ov_metatypes.OVFloorModMetatype,
    ov_metatypes.OVLogicalOrMetatype,
    ov_metatypes.OVLogicalXorMetatype,
    ov_metatypes.OVLogicalAndMetatype,
    ov_metatypes.OVLogicalNotMetatype,
    ov_metatypes.OVPowerMetatype,
    ov_metatypes.OVAvgPoolMetatype,
    ov_metatypes.OVNormalizeL2Metatype,
    ov_metatypes.OVReduceMeanMetatype,
    ov_metatypes.OVInterpolateMetatype,
    ov_metatypes.OVMVNMetatype,
    ov_metatypes.OVLSTMSequenceMetatype,
    ov_metatypes.OVGRUSequenceMetatype,
]


FAKE_QUANTIZE_OPERATIONS = [
    ov_metatypes.OVFakeQuantizeMetatype,
]


CONSTANT_OPERATIONS = [
    ov_metatypes.OVConstantMetatype,
]


SHAPEOF_OPERATIONS = [
    ov_metatypes.OVShapeOfMetatype,
]
