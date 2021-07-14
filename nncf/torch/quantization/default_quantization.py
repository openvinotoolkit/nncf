"""
 Copyright (c) 2021 Intel Corporation
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
from nncf.torch.graph import operator_metatypes
from nncf.torch.graph.operator_metatypes import PTOperatorMetatype

DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT = {
    QuantizationTrait.INPUTS_QUANTIZABLE: [
        operator_metatypes.Conv2dMetatype,
        operator_metatypes.Conv3dMetatype,
        operator_metatypes.ConvTranspose2dMetatype,
        operator_metatypes.ConvTranspose3dMetatype,
        operator_metatypes.DepthwiseConv2dSubtype,
        operator_metatypes.DepthwiseConv3dSubtype,
        operator_metatypes.LinearMetatype,
        operator_metatypes.HardTanhMetatype,
        operator_metatypes.TanhMetatype,
        operator_metatypes.ELUMetatype,
        operator_metatypes.PRELUMetatype,
        operator_metatypes.LeakyRELUMetatype,
        operator_metatypes.LayerNormMetatype,
        operator_metatypes.GELUMetatype,
        operator_metatypes.SigmoidMetatype,
        operator_metatypes.AddMetatype,
        operator_metatypes.MulMetatype,
        operator_metatypes.DivMetatype,
        operator_metatypes.ExpMetatype,
        operator_metatypes.ErfMetatype,
        operator_metatypes.MatMulMetatype,
        operator_metatypes.MeanMetatype,
        operator_metatypes.RoundMetatype,
        operator_metatypes.PixelShuffleMetatype,
        operator_metatypes.BatchNormMetatype,
        operator_metatypes.AvgPool2dMetatype,
        operator_metatypes.AvgPool3dMetatype
    ],
    QuantizationTrait.NON_QUANTIZABLE: [
        operator_metatypes.SoftmaxMetatype
    ],
    QuantizationTrait.CONCAT: [
        operator_metatypes.CatMetatype
    ],
    QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS: [
        operator_metatypes.EmbeddingMetatype,
        operator_metatypes.EmbeddingBagMetatype
    ]
}  # type: Dict[QuantizationTrait, List[PTOperatorMetatype]]
