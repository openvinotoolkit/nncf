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
from typing import Dict, List

from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.torch.graph import operator_metatypes
from nncf.torch.graph.operator_metatypes import OPERATORS_WITH_WEIGHTS_METATYPES
from nncf.torch.graph.operator_metatypes import PTOperatorMetatype

# If a metatype is not in this list, then it is considered to be QuantizationTrait.NON_QUANTIZABLE.

DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT: Dict[QuantizationTrait, List[PTOperatorMetatype]] = {
    QuantizationTrait.INPUTS_QUANTIZABLE: [
        operator_metatypes.PTConv2dMetatype,
        operator_metatypes.PTModuleConv2dMetatype,
        operator_metatypes.PTConv3dMetatype,
        operator_metatypes.PTModuleConv3dMetatype,
        operator_metatypes.PTConvTranspose2dMetatype,
        operator_metatypes.PTModuleConvTranspose2dMetatype,
        operator_metatypes.PTConvTranspose3dMetatype,
        operator_metatypes.PTModuleConvTranspose3dMetatype,
        operator_metatypes.PTDepthwiseConv2dSubtype,
        operator_metatypes.PTDepthwiseConv3dSubtype,
        operator_metatypes.PTModuleDepthwiseConv2dSubtype,
        operator_metatypes.PTModuleDepthwiseConv3dSubtype,
        operator_metatypes.PTLinearMetatype,
        operator_metatypes.PTModuleLinearMetatype,
        operator_metatypes.PTLayerNormMetatype,
        operator_metatypes.PTModuleLayerNormMetatype,
        operator_metatypes.PTAddMetatype,
        operator_metatypes.PTMulMetatype,
        operator_metatypes.PTDivMetatype,
        operator_metatypes.PTMatMulMetatype,
        operator_metatypes.PTMeanMetatype,
        operator_metatypes.PTRoundMetatype,
        operator_metatypes.PTPixelShuffleMetatype,
        operator_metatypes.PTBatchNormMetatype,
        operator_metatypes.PTModuleBatchNormMetatype,
        operator_metatypes.PTAvgPool2dMetatype,
        operator_metatypes.PTAvgPool3dMetatype,
        # 1. Single input activations except Relu and PRelu could not be
        # executed in INT8 precision by the OpenVINO runtime.
        # List of supported operations for INT8 execution:
        # https://docs.openvino.ai/2023.1/openvino_docs_OV_UG_lpt.html#input-model-requirements
        # 2. In case an activation from Torch is fused to
        # a specific OpenVINO operation in runtime, it is better to not quantize
        # this actictivation to keep specific operations fusing.
        # operator_metatypes.PTHardTanhMetatype,
        # operator_metatypes.PTHardSwishMetatype,
        # operator_metatypes.PTHardSigmoidMetatype,
        # operator_metatypes.PTTanhMetatype,
        # operator_metatypes.PTELUMetatype,
        # operator_metatypes.PTLeakyRELUMetatype,
        # operator_metatypes.PTGELUMetatype,
        # operator_metatypes.PTErfMetatype,
        # PTPRELUMetatype is not considered to be QUANTIZATION_AGNOSTIC, because:
        # 1. Runtime doesn't provide performance benefits by quantizing the stand-alone RELU's (ticket: 59548)
        # 2. It's frequently better for the end accuracy to have quantizers set up after the RELU
        # so that the input distribution to the quantizer is non-negative
        # and we can therefore have better quantization resolution while preserving the original dynamic range
        # operator_metatypes.PTPRELUMetatype,
    ],
    QuantizationTrait.QUANTIZATION_AGNOSTIC: [
        operator_metatypes.PTThresholdMetatype,
        operator_metatypes.PTDropoutMetatype,
        operator_metatypes.PTPadMetatype,
        operator_metatypes.PTMaxMetatype,
        operator_metatypes.PTMinMetatype,
        operator_metatypes.PTTransposeMetatype,
        operator_metatypes.PTGatherMetatype,
        operator_metatypes.PTScatterMetatype,
        operator_metatypes.PTReshapeMetatype,
        operator_metatypes.PTSqueezeMetatype,
        operator_metatypes.PTSplitMetatype,
        operator_metatypes.PTExpandMetatype,
        operator_metatypes.PTMaxPool1dMetatype,
        operator_metatypes.PTMaxPool2dMetatype,
        operator_metatypes.PTMaxPool3dMetatype,
        operator_metatypes.PTMaxUnpool1dMetatype,
        operator_metatypes.PTMaxUnpool2dMetatype,
        operator_metatypes.PTMaxUnpool3dMetatype,
        operator_metatypes.PTRepeatMetatype,
        operator_metatypes.PTNoopMetatype,
        # PTRELUMetatype is not considered to be QUANTIZATION_AGNOSTIC, because:
        # 1. Runtime doesn't provide performance benefits by quantizing the stand-alone RELU's (ticket: 59548)
        # 2. It's frequently better for the end accuracy to have quantizers set up after the RELU
        # so that the input distribution to the quantizer is non-negative
        # and we can therefore have better quantization resolution while preserving the original dynamic range
    ],
    QuantizationTrait.CONCAT: [operator_metatypes.PTCatMetatype],
    QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS: [
        operator_metatypes.PTEmbeddingMetatype,
        operator_metatypes.PTModuleEmbeddingMetatype,
        operator_metatypes.PTEmbeddingBagMetatype,
        operator_metatypes.PTModuleEmbeddingBagMetatype,
    ],
}


QUANTIZATION_LAYER_METATYPES: List[PTOperatorMetatype] = OPERATORS_WITH_WEIGHTS_METATYPES
