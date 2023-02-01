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
from typing import Dict, List

from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.torch.graph import operator_metatypes
from nncf.torch.graph.operator_metatypes import PTOperatorMetatype
from nncf.torch.graph.operator_metatypes import OPERATORS_WITH_WEIGHTS_METATYPES

# If there are no some metatypes it means that they are considered as QuantizationTrait.QuantizationAgnostic

DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT = {
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
        operator_metatypes.PTLinearMetatype,
        operator_metatypes.PTModuleLinearMetatype,
        operator_metatypes.PTHardTanhMetatype,
        operator_metatypes.PTHardSwishMetatype,
        operator_metatypes.PTHardSigmoidMetatype,
        operator_metatypes.PTTanhMetatype,
        operator_metatypes.PTELUMetatype,
        operator_metatypes.PTPRELUMetatype,
        operator_metatypes.PTLeakyRELUMetatype,
        operator_metatypes.PTLayerNormMetatype,
        operator_metatypes.PTModuleLayerNormMetatype,
        operator_metatypes.PTGELUMetatype,
        operator_metatypes.PTAddMetatype,
        operator_metatypes.PTMulMetatype,
        operator_metatypes.PTDivMetatype,
        operator_metatypes.PTErfMetatype,
        operator_metatypes.PTMatMulMetatype,
        operator_metatypes.PTMeanMetatype,
        operator_metatypes.PTRoundMetatype,
        operator_metatypes.PTPixelShuffleMetatype,
        operator_metatypes.PTBatchNormMetatype,
        operator_metatypes.PTModuleBatchNormMetatype,
        operator_metatypes.PTAvgPool2dMetatype,
        operator_metatypes.PTAvgPool3dMetatype
    ],
    QuantizationTrait.NON_QUANTIZABLE: [
        operator_metatypes.PTSigmoidMetatype,
        operator_metatypes.PTExpMetatype,
        operator_metatypes.PTSoftmaxMetatype,
        UnknownMetatype
    ],
    QuantizationTrait.CONCAT: [
        operator_metatypes.PTCatMetatype
    ],
    QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS: [
        operator_metatypes.PTEmbeddingMetatype,
        operator_metatypes.PTModuleEmbeddingMetatype,
        operator_metatypes.PTEmbeddingBagMetatype,
        operator_metatypes.PTModuleEmbeddingBagMetatype
    ]
}  # type: Dict[QuantizationTrait, List[PTOperatorMetatype]]


QUANTIZATION_LAYER_METATYPES = OPERATORS_WITH_WEIGHTS_METATYPES # type: List[PTOperatorMetatype]
