# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.torch.graph import operator_metatypes as om

# If a metatype is not in this list, then it is considered to be QuantizationTrait.NON_QUANTIZABLE.

DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT: dict[QuantizationTrait, list[type[om.PTOperatorMetatype]]] = {
    QuantizationTrait.INPUTS_QUANTIZABLE: [
        om.PTConv2dMetatype,
        om.PTConv3dMetatype,
        om.PTConvTranspose2dMetatype,
        om.PTConvTranspose3dMetatype,
        om.PTDepthwiseConv2dSubtype,
        om.PTDepthwiseConv3dSubtype,
        om.PTLinearMetatype,
        om.PTLayerNormMetatype,
        om.PTAddMetatype,
        om.PTMulMetatype,
        om.PTDivMetatype,
        om.PTMatMulMetatype,
        om.PTMeanMetatype,
        om.PTRoundMetatype,
        om.PTPixelShuffleMetatype,
        om.PTBatchNormMetatype,
        om.PTAvgPool2dMetatype,
        om.PTAvgPool3dMetatype,
        # 1. Single input activations except Relu and PRelu could not be
        # executed in INT8 precision by the OpenVINO runtime.
        # List of supported operations for INT8 execution:
        # https://docs.openvino.ai/2023.1/openvino_docs_OV_UG_lpt.html#input-model-requirements
        # 2. In case an activation from Torch is fused to
        # a specific OpenVINO operation in runtime, it is better to not quantize
        # this activation to keep specific operations fusing.
        # om.PTHardTanhMetatype,
        # om.PTHardSwishMetatype,
        # om.PTHardSigmoidMetatype,
        # om.PTTanhMetatype,
        # om.PTELUMetatype,
        # om.PTLeakyRELUMetatype,
        # om.PTGELUMetatype,
        # om.PTErfMetatype,
        # PTPRELUMetatype is not considered to be QUANTIZATION_AGNOSTIC, because:
        # 1. Runtime doesn't provide performance benefits by quantizing the stand-alone RELU's (ticket: 59548)
        # 2. It's frequently better for the end accuracy to have quantizers set up after the RELU
        # so that the input distribution to the quantizer is non-negative
        # and we can therefore have better quantization resolution while preserving the original dynamic range
        # om.PTPRELUMetatype,
    ],
    QuantizationTrait.QUANTIZATION_AGNOSTIC: [
        om.PTThresholdMetatype,
        om.PTDropoutMetatype,
        om.PTPadMetatype,
        om.PTMaxMetatype,
        om.PTMinMetatype,
        om.PTTransposeMetatype,
        om.PTGatherMetatype,
        om.PTScatterMetatype,
        om.PTReshapeMetatype,
        om.PTSqueezeMetatype,
        om.PTSplitMetatype,
        om.PTExpandMetatype,
        om.PTMaxPool1dMetatype,
        om.PTMaxPool2dMetatype,
        om.PTMaxPool3dMetatype,
        om.PTMaxUnpool1dMetatype,
        om.PTMaxUnpool2dMetatype,
        om.PTMaxUnpool3dMetatype,
        om.PTRepeatMetatype,
        om.PTNoopMetatype,
        # PTRELUMetatype is not considered to be QUANTIZATION_AGNOSTIC, because:
        # 1. Runtime doesn't provide performance benefits by quantizing the stand-alone RELU's (ticket: 59548)
        # 2. It's frequently better for the end accuracy to have quantizers set up after the RELU
        # so that the input distribution to the quantizer is non-negative
        # and we can therefore have better quantization resolution while preserving the original dynamic range
    ],
    QuantizationTrait.CONCAT: [om.PTCatMetatype],
    QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS: [
        om.PTEmbeddingMetatype,
        om.PTEmbeddingBagMetatype,
    ],
}
