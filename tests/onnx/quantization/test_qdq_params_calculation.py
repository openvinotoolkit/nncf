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

import numpy as np
import onnx
import pytest

from nncf.common.quantization.structs import QuantizationPreset
from nncf.onnx.graph.onnx_helper import get_tensor_value
from nncf.onnx.quantization.quantizer_parameters import convert_fc_params_to_onnx_params
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import FP8Type
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.fake_quantize import FakeConvertParameters
from nncf.tensor import Tensor
from tests.cross_fw.shared.comparator import compare_stats
from tests.cross_fw.shared.json import load_json
from tests.onnx.conftest import ONNX_TEST_ROOT
from tests.onnx.models import EmbeddingModel
from tests.onnx.models import GEMMTransposeWeightModel
from tests.onnx.models import LinearModel
from tests.onnx.models import MatMulActivationModel
from tests.onnx.models import MatMulWeightModel
from tests.onnx.models import OneDepthwiseConvolutionalModel
from tests.onnx.models import ReshapeWeightModel
from tests.onnx.models import WeightSharingModel
from tests.onnx.quantization.common import min_max_quantize_model

REFERENCE_SCALES_DIR = ONNX_TEST_ROOT / "data" / "reference_scales"


def get_q_nodes_params(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    output = {}
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            scale = get_tensor_value(model, node.input[1])
            zero_point = get_tensor_value(model, node.input[2])
            output[node.name] = {"scale": scale, "zero_point": zero_point}
    return output


@pytest.mark.parametrize(
    "overflow_fix",
    [OverflowFix.DISABLE, OverflowFix.ENABLE, OverflowFix.FIRST_LAYER],
    ids=[OverflowFix.DISABLE.value, OverflowFix.ENABLE.value, OverflowFix.FIRST_LAYER.value],
)
def test_overflow_fix_scales(overflow_fix):
    model = LinearModel()
    quantized_model = min_max_quantize_model(
        model.onnx_model,
        quantization_params={"advanced_parameters": AdvancedQuantizationParameters(overflow_fix=overflow_fix)},
    )
    q_nodes_params = get_q_nodes_params(quantized_model)

    ref_stats_name = model.path_ref_graph.split(".")[0] + f"_overflow_fix_{overflow_fix.value}.json"
    ref_stats_path = REFERENCE_SCALES_DIR / ref_stats_name

    # Unkomment lines below to generate reference for new models.
    # from tests.cross_fw.shared.json import dump_to_json

    # dump_to_json(ref_stats_path, q_nodes_params)

    ref_nodes_params = load_json(ref_stats_path)
    compare_stats(ref_nodes_params, q_nodes_params)


MODELS = [
    GEMMTransposeWeightModel,
    MatMulWeightModel,
    MatMulActivationModel,
    WeightSharingModel,
    ReshapeWeightModel,
    LinearModel,
    OneDepthwiseConvolutionalModel,
    EmbeddingModel,
]


@pytest.mark.parametrize(
    "preset",
    [QuantizationPreset.PERFORMANCE, QuantizationPreset.MIXED],
    ids=[QuantizationPreset.PERFORMANCE.value, QuantizationPreset.MIXED.value],
)
@pytest.mark.parametrize(
    "model",
    MODELS,
    ids=[
        "GEMMTransposeWeightModel",
        "MatMulWeightModel",
        "MatMulActivationModel",
        "WeightSharingModel",
        "ReshapeWeightModel",
        "LinearModel",
        "OneDepthwiseConvolutionalModel",
        "EmbeddingModel",
    ],
)
def test_scales(model, preset):
    model = model()
    quantized_model = min_max_quantize_model(model.onnx_model, quantization_params={"preset": preset})
    q_nodes_params = get_q_nodes_params(quantized_model)

    ref_stats_name = model.path_ref_graph.split(".")[0] + f"_{preset.value}.json"
    ref_stats_path = REFERENCE_SCALES_DIR / ref_stats_name

    # Unkomment lines below to generate reference for new models.
    # from tests.cross_fw.shared.json import dump_to_json
    # dump_to_json(ref_stats_path, q_nodes_params)

    ref_nodes_params = load_json(ref_stats_path)
    compare_stats(ref_nodes_params, q_nodes_params)


@pytest.mark.parametrize(
    "scale,shift,destination_type,axis,expected",
    [
        (
            np.array(2, dtype=np.float32),
            np.array(0, dtype=np.float32),
            FP8Type.E4M3,
            None,
            (np.array(0.5, dtype=np.float32), np.array(0, dtype=np.float32), onnx.TensorProto.FLOAT8E4M3FN),
        ),
        (
            np.array(2, dtype=np.float32),
            np.array(0, dtype=np.float32),
            FP8Type.E5M2,
            None,
            (np.array(0.5, dtype=np.float32), np.array(0, dtype=np.float32), onnx.TensorProto.FLOAT8E5M2),
        ),
        (
            np.array(0, dtype=np.float32),
            np.array(0, dtype=np.float32),
            FP8Type.E5M2,
            None,
            (np.array(8388608, dtype=np.float32), np.array(0, dtype=np.float32), onnx.TensorProto.FLOAT8E5M2),
        ),
        (
            np.array([2, 4, 8], dtype=np.float32).reshape(3, 1, 1, 1),
            np.array([0, 0, 0], dtype=np.float32).reshape(3, 1, 1, 1),
            FP8Type.E4M3,
            0,
            (
                np.array([0.5, 0.25, 0.125], dtype=np.float32),
                np.array([0, 0, 0], dtype=np.float32),
                onnx.TensorProto.FLOAT8E4M3FN,
            ),
        ),
    ],
)
def test_convert_fc_params_to_onnx_params(scale, shift, destination_type, axis, expected):
    fc_params = FakeConvertParameters(Tensor(scale), Tensor(shift), destination_type)
    onnx_params = convert_fc_params_to_onnx_params(fc_params, axis)
    expected_scale, expected_zp, expected_type = expected

    assert expected_scale.shape == onnx_params.scale.shape
    assert np.allclose(expected_scale, onnx_params.scale, atol=1e-5)

    assert expected_zp.shape == onnx_params.zero_point.shape
    assert np.allclose(expected_zp, onnx_params.zero_point, atol=1e-5)

    assert expected_type == onnx_params.tensor_type
    assert axis == onnx_params.axis
