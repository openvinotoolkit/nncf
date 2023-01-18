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

from nncf.experimental.openvino_native.graph.metatypes import openvino_metatypes as ov_metatypes


QUANTIZE_AGNOSTIC_OPERATIONS = [
    ov_metatypes.OVMaxPoolMetatype,
    ov_metatypes.OVReduceMaxMetatype,
    ov_metatypes.OVReshapeMetatype,
    ov_metatypes.OVConcatMetatype,
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
    ov_metatypes.OVDepthToSpaceMetatype,
]


QUANTIZABLE_OPERATIONS = [
    ov_metatypes.OVConvolutionMetatype,
    ov_metatypes.OVConvolutionBackpropDataMetatype,
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
