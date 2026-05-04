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


from nncf.common.hardware.defines import ADJUST_PADDING
from nncf.common.hardware.defines import SCALES
from nncf.common.hardware.defines import UNIFIED
from nncf.common.hardware.defines import OpDesc
from nncf.common.hardware.opset import HWOpName
from nncf.common.hardware.qspaces import q2_w
from nncf.common.hardware.qspaces import q4_tn
from nncf.common.hardware.qspaces import q4_w
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

NPU_SETUP: tuple[OpDesc, ...] = (
    OpDesc(
        type=HWOpName.CONVOLUTION,
        activations=(q8_a, q16_a, q4_tn),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym, q4_w, q4_tn),
        attributes={ADJUST_PADDING: True},
    ),
    OpDesc(
        type=HWOpName.DEPTHWISE_CONVOLUTION,
        activations=(q8_a_ch, q16_a_ch),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym, q4_w, q2_w),
    ),
    OpDesc(
        type=HWOpName.MAT_MUL,
        activations=(q8_a, q16_a, q4_tn),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym, q4_w, q2_w),
    ),
    OpDesc(
        type=HWOpName.ADD,
        activations=(q8_a, q16_a, q4_tn),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym, q4_tn),
        attributes={SCALES: UNIFIED},
    ),
    OpDesc(
        type=HWOpName.MULTIPLY,
        activations=(q8_a, q16_a, q4_tn),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym, q4_tn),
        attributes={SCALES: UNIFIED},
    ),
    OpDesc(
        type=HWOpName.MAXIMUM,
        activations=(q8_a, q16_a, q4_tn),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym, q4_tn),
        attributes={SCALES: UNIFIED},
    ),
    OpDesc(
        type=HWOpName.LESS,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
        attributes={SCALES: UNIFIED},
    ),
    OpDesc(
        type=HWOpName.LESS_EQUAL,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
        attributes={SCALES: UNIFIED},
    ),
    OpDesc(
        type=HWOpName.GREATER,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
        attributes={SCALES: UNIFIED},
    ),
    OpDesc(
        type=HWOpName.GREATER_EQUAL,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
        attributes={SCALES: UNIFIED},
    ),
    OpDesc(
        type=HWOpName.DIVIDE,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
        attributes={SCALES: UNIFIED},
    ),
    OpDesc(
        type=HWOpName.MINIMUM,
        activations=(q8_a, q16_a, q4_tn),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym, q4_tn),
        attributes={SCALES: UNIFIED},
    ),
    OpDesc(
        type=HWOpName.EQUAL,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
        attributes={SCALES: UNIFIED},
    ),
    OpDesc(
        type=HWOpName.SUBTRACT,
        activations=(q8_a, q16_a, q4_tn),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym, q4_tn),
        attributes={SCALES: UNIFIED},
    ),
    OpDesc(
        type=HWOpName.NOT_EQUAL,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
        attributes={SCALES: UNIFIED},
    ),
    OpDesc(
        type=HWOpName.FLOOR_MOD,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
        attributes={SCALES: UNIFIED},
    ),
    OpDesc(
        type=HWOpName.LOGICAL_OR,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
        attributes={SCALES: UNIFIED},
    ),
    OpDesc(
        type=HWOpName.LOGICAL_XOR,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
        attributes={SCALES: UNIFIED},
    ),
    OpDesc(
        type=HWOpName.LOGICAL_AND,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
        attributes={SCALES: UNIFIED},
    ),
    OpDesc(
        type=HWOpName.LOGICAL_NOT,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type=HWOpName.POWER,
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type=HWOpName.AVG_POOL,
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type=HWOpName.NORMALIZE_L2,
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type=HWOpName.REDUCE_L2,
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type=HWOpName.REDUCE_MEAN,
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type=HWOpName.INTERPOLATE,
        activations=(q8_a, q16_a),
        attributes={"mode": "linear"},
    ),
    OpDesc(
        type=HWOpName.INTERPOLATE,
        attributes={"mode": "nearest"},
    ),
    OpDesc(
        type=HWOpName.MVN,
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type=HWOpName.CONCAT,
        attributes={SCALES: UNIFIED},
    ),
    OpDesc(
        type=HWOpName.LSTM_SEQUENCE,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q16_w_sym),
    ),
    OpDesc(
        type=HWOpName.GRU_SEQUENCE,
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q16_w_sym),
    ),
    OpDesc(
        type=HWOpName.REDUCE_SUM,
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type=HWOpName.GROUP_NORMALIZATION,
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type=HWOpName.SCALED_DOT_PRODUCT_ATTENTION,
        activations=(q8_a_sym, q16_a_sym),
    ),
    OpDesc(type=HWOpName.MAX_POOL),
    OpDesc(type=HWOpName.REDUCE_MAX),
    OpDesc(type=HWOpName.RESHAPE),
    OpDesc(type=HWOpName.FLATTEN),
    OpDesc(type=HWOpName.SQUEEZE),
    OpDesc(type=HWOpName.UNSQUEEZE),
    OpDesc(type=HWOpName.SPLIT),
    OpDesc(type=HWOpName.VARIADIC_SPLIT),
    OpDesc(type=HWOpName.CROP),
    OpDesc(type=HWOpName.TRANSPOSE),
    OpDesc(type=HWOpName.TILE),
    OpDesc(type=HWOpName.STRIDED_SLICE),
    OpDesc(type=HWOpName.SLICE),
    OpDesc(type=HWOpName.SHUFFLE_CHANNELS),
    OpDesc(type=HWOpName.BROADCAST),
    OpDesc(type=HWOpName.PAD),
    OpDesc(type=HWOpName.CONVERT_LIKE),
    OpDesc(
        type=HWOpName.EMBEDDING,
        weights=(q8_w_sym_any_nr, q8_w_asym, q16_w_sym_any_nr, q16_w_asym),
    ),
    OpDesc(
        type=HWOpName.EMBEDDING_BAG,
        weights=(q8_w_sym_any_nr, q8_w_asym, q16_w_sym_any_nr, q16_w_asym),
    ),
)
