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

from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVAddMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVAndMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVAveragePoolMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVBatchNormMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVConcatMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVConvolutionBackpropDataMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVDivMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVEqualMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVExpMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVFloorModMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVGreaterMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVGreaterEqualMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVInterpolateMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVLessMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVLessEqualMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVLogMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVMVNMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVMaximumMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVMinimumMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVMulMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVNormalizeL2Metatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVNotMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVNotEqualMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVOrMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVPowerMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVReduceMeanMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVSigmoidMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVSoftmaxMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVSubMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVXorMetatype


DEFAULT_OV_QUANT_TRAIT_TO_OP_DICT = {
    QuantizationTrait.INPUTS_QUANTIZABLE: [
        OVConvolutionMetatype,
        OVConvolutionBackpropDataMetatype,
        OVMatMulMetatype,
        OVBatchNormMetatype,
        OVAddMetatype,
        OVSubMetatype,
        OVMulMetatype,
        OVDivMetatype,
        OVMaximumMetatype,
        OVMinimumMetatype,
        OVAveragePoolMetatype,
        OVReduceMeanMetatype,
        OVMVNMetatype,
        OVNormalizeL2Metatype,
        OVInterpolateMetatype,
        OVPowerMetatype,
        OVFloorModMetatype,
        OVLessMetatype,
        OVLessEqualMetatype,
        OVGreaterMetatype,
        OVGreaterEqualMetatype,
        OVEqualMetatype,
        OVNotEqualMetatype,
        OVNotMetatype,
        OVAndMetatype,
        OVOrMetatype,
        OVXorMetatype,
    ],
    QuantizationTrait.NON_QUANTIZABLE: [OVSigmoidMetatype,
                                        OVSoftmaxMetatype,
                                        OVExpMetatype,
                                        OVLogMetatype,
                                        UnknownMetatype],
    QuantizationTrait.CONCAT: [OVConcatMetatype],
    QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS: []
}
