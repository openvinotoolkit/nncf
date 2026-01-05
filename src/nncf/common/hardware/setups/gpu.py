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


from nncf.common.hardware.defines import SCALES
from nncf.common.hardware.defines import UNIFIED
from nncf.common.hardware.defines import OpDesc
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
        type="Convolution",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="DepthWiseConvolution",
        activations=(q8_a_ch, q16_a_ch),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="MatMul",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="Add",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="Multiply",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="Maximum",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="Less",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="LessEqual",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="Greater",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="GreaterEqual",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="Divide",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="Minimum",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="Equal",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="Subtract",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="NotEqual",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="FloorMod",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="LogicalOr",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="LogicalXor",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="LogicalAnd",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="LogicalNot",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q8_w_asym, q16_w_sym, q16_w_asym),
    ),
    OpDesc(
        type="Power",
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type="AvgPool",
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type="NormalizeL2",
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type="ReduceL2",
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type="ReduceMean",
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type="Interpolate",
        activations=(q8_a, q16_a),
        attributes={"mode": "linear"},
    ),
    OpDesc(
        type="Interpolate",
        attributes={"mode": "nearest"},
    ),
    OpDesc(
        type="MVN",
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type="Concat",
        attributes={SCALES: UNIFIED},
    ),
    OpDesc(
        type="LSTMSequence",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q16_w_sym),
    ),
    OpDesc(
        type="GRUSequence",
        activations=(q8_a, q16_a),
        weights=(q8_w_sym, q16_w_sym),
    ),
    OpDesc(
        type="ReduceSum",
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type="GroupNormalization",
        activations=(q8_a, q16_a),
    ),
    OpDesc(
        type="ScaledDotProductAttention",
        activations=(q8_a_sym, q16_a_sym),
    ),
    OpDesc(type="MaxPool"),
    OpDesc(type="ReduceMax"),
    OpDesc(type="Reshape"),
    OpDesc(type="Flatten"),
    OpDesc(type="Squeeze"),
    OpDesc(type="Unsqueeze"),
    OpDesc(type="Split"),
    OpDesc(type="VariadicSplit"),
    OpDesc(type="Crop"),
    OpDesc(type="Transpose"),
    OpDesc(type="Tile"),
    OpDesc(type="StridedSlice"),
    OpDesc(type="Slice"),
    OpDesc(type="ShuffleChannels"),
    OpDesc(type="Broadcast"),
    OpDesc(type="Pad"),
    OpDesc(type="ConvertLike"),
    OpDesc(
        type="Embedding",
        weights=(q8_w_sym_any_nr, q8_w_asym, q16_w_sym_any_nr, q16_w_asym),
    ),
    OpDesc(type="EmbeddingBag"),
)
