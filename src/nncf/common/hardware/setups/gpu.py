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


from nncf.common.hardware.defines import SCALES
from nncf.common.hardware.defines import UNIFIED
from nncf.common.hardware.defines import OpDesc
from nncf.common.hardware.opset import HWConfigOpName
from nncf.common.hardware.qspaces import q8_a
from nncf.common.hardware.qspaces import q8_a_ch
from nncf.common.hardware.qspaces import q8_a_sym
from nncf.common.hardware.qspaces import q8_w_asym
from nncf.common.hardware.qspaces import q8_w_sym
from nncf.common.hardware.qspaces import q8_w_sym_any_nr
from nncf.common.hardware.qspaces import q16_a
from nncf.common.hardware.qspaces import q16_a_ch
from nncf.common.hardware.qspaces import q16_a_sym
from nncf.common.hardware.qspaces import q16_w_asym
from nncf.common.hardware.qspaces import q16_w_sym
from nncf.common.hardware.qspaces import q16_w_sym_any_nr

GPU_SETUP: tuple[OpDesc, ...] = (
    OpDesc(
        type=HWConfigOpName.CONVOLUTION,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.DEPTHWISECONVOLUTION,
        activations=(q8_a_ch, q16_a_ch),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.MATMUL,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.ADD,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.MULTIPLY,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.MAXIMUM,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.LESS,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.LESSEQUAL,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.GREATER,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.GREATEREQUAL,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.DIVIDE,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.MINIMUM,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.EQUAL,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.SUBTRACT,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.NOTEQUAL,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.FLOORMOD,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.LOGICALOR,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.LOGICALXOR,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.LOGICALAND,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.LOGICALNOT,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWConfigOpName.POWER,
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type=HWConfigOpName.AVGPOOL,
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type=HWConfigOpName.NORMALIZEL2,
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type=HWConfigOpName.REDUCEL2,
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type=HWConfigOpName.REDUCEMEAN,
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type=HWConfigOpName.INTERPOLATE,
        activations=(q8_a, q16_a),
        attributes={"mode": "linear"},
    ),
    OpDesc(
        type=HWConfigOpName.INTERPOLATE,
        attributes={"mode": "nearest"},
    ),
    OpDesc(
        type=HWConfigOpName.MVN,
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type=HWConfigOpName.CONCAT,
        attributes={SCALES: UNIFIED},
    ),
    OpDesc(
        type=HWConfigOpName.LSTMSEQUENCE,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q16_w_sym),
    ),
    OpDesc(
        type=HWConfigOpName.GRUSEQUENCE,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q16_w_sym),
    ),
    OpDesc(
        type=HWConfigOpName.REDUCESUM,
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type=HWConfigOpName.GROUPNORMALIZATION,
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type=HWConfigOpName.SCALED_DOT_PRODUCT_ATTENTION,
        activations=(q8_a_sym, q16_a_sym),
    ),
    OpDesc(type=HWConfigOpName.MAXPOOL),
    OpDesc(type=HWConfigOpName.REDUCEMAX),
    OpDesc(type=HWConfigOpName.RESHAPE),
    OpDesc(type=HWConfigOpName.FLATTEN),
    OpDesc(type=HWConfigOpName.SQUEEZE),
    OpDesc(type=HWConfigOpName.UNSQUEEZE),
    OpDesc(type=HWConfigOpName.SPLIT),
    OpDesc(type=HWConfigOpName.VARIADICSPLIT),
    OpDesc(type=HWConfigOpName.CROP),
    OpDesc(type=HWConfigOpName.TRANSPOSE),
    OpDesc(type=HWConfigOpName.TILE),
    OpDesc(type=HWConfigOpName.STRIDEDSLICE),
    OpDesc(type=HWConfigOpName.SLICE),
    OpDesc(type=HWConfigOpName.SHUFFLECHANNELS),
    OpDesc(type=HWConfigOpName.BROADCAST),
    OpDesc(type=HWConfigOpName.PAD),
    OpDesc(type=HWConfigOpName.CONVERTLIKE),
    OpDesc(
        type=HWConfigOpName.EMBEDDING,
        weights=(q8_w_sym_any_nr, q8_w_asym, q16_w_sym_any_nr, q16_w_asym),
    ),
    OpDesc(type=HWConfigOpName.EMBEDDINGBAG),
)
