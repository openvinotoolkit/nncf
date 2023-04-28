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

from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.openvino.graph.metatypes import openvino_metatypes as ov_metatypes

DEFAULT_OV_QUANT_TRAIT_TO_OP_DICT = {
    QuantizationTrait.INPUTS_QUANTIZABLE: [
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
    ],
    QuantizationTrait.NON_QUANTIZABLE: [
        ov_metatypes.OVSigmoidMetatype,
        ov_metatypes.OVSoftmaxMetatype,
        ov_metatypes.OVAssignMetatype,
        ov_metatypes.OVDeformableConvolutionMetatype,
        UnknownMetatype,
        # Ticket: 108478
        ov_metatypes.OVReluMetatype,
        ov_metatypes.OVLogMetatype,
        ov_metatypes.OVExpMetatype,
        ov_metatypes.OVSqrtMetatype,
        ov_metatypes.OVAbsMetatype,
    ],
    QuantizationTrait.CONCAT: [ov_metatypes.OVConcatMetatype],
    QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS: [],
}
