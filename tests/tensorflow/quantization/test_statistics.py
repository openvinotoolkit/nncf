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

from typing import List, Optional

import pytest

from nncf.common.quantization.statistics import QuantizationStatistics
from nncf.common.quantization.statistics import QuantizersCounter
from tests.tensorflow import test_models
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.helpers import get_empty_config


def _get_basic_quantization_config(mode: str, granularity: str, input_sample_sizes: Optional[List[int]] = None):
    config = get_empty_config(input_sample_sizes)
    per_channel = granularity == "per_channel"

    compression_section = {
        "algorithm": "quantization",
        "activations": {
            "mode": mode,
            "per_channel": per_channel,
        },
        "weights": {
            "mode": mode,
            "per_channel": per_channel,
        },
    }

    config["compression"] = compression_section
    config["target_device"] = "TRIAL"
    return config


class Case:
    def __init__(
        self,
        model_name: str,
        model_builder,
        input_sample_sizes: List[int],
        mode: str,
        granularity: str,
        expected: QuantizationStatistics,
    ):
        self._model_name = model_name
        self._model_builder = model_builder
        self._input_sample_sizes = input_sample_sizes
        self._mode = mode
        self._granularity = granularity
        self._expected = expected

    @property
    def model(self):
        return self._model_builder(input_shape=tuple(self._input_sample_sizes[1:]))

    @property
    def config(self):
        return _get_basic_quantization_config(self._mode, self._granularity, self._input_sample_sizes)

    @property
    def expected(self):
        return self._expected

    def get_id(self) -> str:
        return f"{self._model_name}-{self._mode}-{self._granularity}"


TEST_CASES = [
    Case(
        model_name="mobilenet_v2",
        model_builder=test_models.MobileNetV2,
        input_sample_sizes=[1, 96, 96, 3],
        mode="symmetric",
        granularity="per_tensor",
        expected=QuantizationStatistics(
            wq_counter=QuantizersCounter(53, 0, 53, 0, 53, 0, 53),
            aq_counter=QuantizersCounter(64, 0, 64, 0, 64, 0, 64),
            num_wq_per_bitwidth={8: 53},
            num_aq_per_bitwidth={8: 64},
            ratio_of_enabled_quantizations=100.0,
        ),
    ),
    Case(
        model_name="mobilenet_v2",
        model_builder=test_models.MobileNetV2,
        input_sample_sizes=[1, 96, 96, 3],
        mode="asymmetric",
        granularity="per_channel",
        expected=QuantizationStatistics(
            wq_counter=QuantizersCounter(0, 53, 53, 0, 0, 53, 53),
            aq_counter=QuantizersCounter(0, 64, 64, 0, 0, 64, 64),
            num_wq_per_bitwidth={8: 53},
            num_aq_per_bitwidth={8: 64},
            ratio_of_enabled_quantizations=100.0,
        ),
    ),
]
TEST_CASES_IDS = [test_case.get_id() for test_case in TEST_CASES]


@pytest.mark.parametrize("test_case", TEST_CASES, ids=TEST_CASES_IDS)
def test_quantization_statistics(test_case):
    _, compression_ctrl = create_compressed_model_and_algo_for_test(
        test_case.model, test_case.config, force_no_init=True
    )
    actual = compression_ctrl.statistics().quantization
    expected = test_case.expected

    assert expected.wq_counter.__dict__ == actual.wq_counter.__dict__
    assert expected.aq_counter.__dict__ == actual.aq_counter.__dict__
    assert expected.num_wq_per_bitwidth == actual.num_wq_per_bitwidth
    assert expected.num_aq_per_bitwidth == actual.num_aq_per_bitwidth
    assert expected.ratio_of_enabled_quantizations == actual.ratio_of_enabled_quantizations


def test_full_ignored_scope():
    shape = [1, 96, 96, 3]
    config = _get_basic_quantization_config("symmetric", "per_tensor", shape)
    config["compression"]["ignored_scopes"] = ["{re}.*"]
    model = test_models.MobileNetV2(input_shape=tuple(shape[1:]))
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config, force_no_init=True)
    compression_ctrl.statistics()
