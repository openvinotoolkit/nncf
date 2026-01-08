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
from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns import merge_two_types_of_operations
from nncf.torch.graph import operator_metatypes as om

LINEAR_OPERATIONS = {
    GraphPattern.METATYPE_ATTR: [
        # Linear
        om.PTLinearMetatype,
        # Conv1D
        om.PTConv1dMetatype,
        om.PTDepthwiseConv1dSubtype,
        # Conv2D
        om.PTConv2dMetatype,
        om.PTDepthwiseConv2dSubtype,
        # Conv3D
        om.PTConv3dMetatype,
        om.PTDepthwiseConv3dSubtype,
        # Transposed conv
        om.PTConvTranspose1dMetatype,
        om.PTConvTranspose2dMetatype,
        om.PTConvTranspose3dMetatype,
        # Deform conv
        om.PTDeformConv2dMetatype,
        # MatMul
        om.PTMatMulMetatype,
        # Addmm
        om.PTAddmmMetatype,
    ],
    GraphPattern.LABEL_ATTR: "LINEAR",
}

BATCH_NORMALIZATION_OPERATIONS = {
    GraphPattern.METATYPE_ATTR: [om.PTBatchNormMetatype],
    GraphPattern.LABEL_ATTR: "BATCH_NORMALIZATION",
}

GROUP_NORMALIZATION_OPERATIONS = {
    GraphPattern.METATYPE_ATTR: [om.PTGroupNormMetatype],
    GraphPattern.LABEL_ATTR: "GROUP_NORMALIZATION",
}

LAYER_NORMALIZATION_OPERATIONS = {
    GraphPattern.METATYPE_ATTR: [om.PTLayerNormMetatype],
    GraphPattern.LABEL_ATTR: "LAYER_NORMALIZATION",
}

RELU_OPERATIONS = {
    GraphPattern.METATYPE_ATTR: [
        om.PTRELUMetatype,
        om.PTHardTanhMetatype,
    ],
    GraphPattern.LABEL_ATTR: "RELU",
}

NON_RELU_ACTIVATIONS_OPERATIONS = {
    GraphPattern.METATYPE_ATTR: [
        om.PTELUMetatype,
        om.PTPRELUMetatype,
        om.PTLeakyRELUMetatype,
        om.PTSigmoidMetatype,
        om.PTGELUMetatype,
        om.PTSILUMetatype,
        om.PTHardSigmoidMetatype,
        om.PTHardSwishMetatype,
        om.PTSELUMetatype,
    ],
    GraphPattern.LABEL_ATTR: "NON_RELU_ACTIVATIONS",
}

ATOMIC_ACTIVATIONS_OPERATIONS = merge_two_types_of_operations(
    RELU_OPERATIONS, NON_RELU_ACTIVATIONS_OPERATIONS, "ATOMIC_ACTIVATIONS"
)

ARITHMETIC_OPERATIONS = {
    GraphPattern.METATYPE_ATTR: [om.PTAddMetatype, om.PTSubMetatype, om.PTMulMetatype, om.PTDivMetatype],
    GraphPattern.LABEL_ATTR: "ARITHMETIC",
}
