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
import math
from abc import ABC
from abc import abstractmethod
from typing import TypeVar

import numpy as np
import pytest

from nncf import CompressWeightsMode
from nncf import SensitivityMetric
from nncf.data.dataset import Dataset
from nncf.quantization import compress_weights
from nncf.quantization.algorithms.weight_compression.mixed_precision import MIXED_PRECISION_CRITERIA
from nncf.tensor import TensorDataType

TTensor = TypeVar("TTensor")

NON_ZERO_ROW = [-4, 1, 2]
ACTIVATION = [[NON_ZERO_ROW, [0, 0, 0], [0, 0, 0]]]
MAX_VAR = 3.555555  # np.max(np.var(ACTIVATION, 1))
MEAN_VAR = 1.555555  # np.mean(np.var(ACTIVATION, 1))
MEAN_MAX = 2.333333  # np.mean(np.max(np.abs(ACTIVATION), 1))
HESSIAN_TRACE = (16 + 1 + 4) * 2 / 9  # sum(i*i for i in NON_ZERO_ROW) * 2 / ACTIVATION.size
MAX_BASELINE_SCORE = 1 / 1.1920928955078125e-07


class TemplateWeightCompression(ABC):
    @staticmethod
    @abstractmethod
    def cast_to(x: TTensor, dtype: TensorDataType) -> TTensor:
        pass

    @abstractmethod
    def get_matmul_model(self):
        """Returns a model instance."""

    @pytest.mark.parametrize(
        ("mode", "ref_act_score", "ref_score"),
        (
            (SensitivityMetric.HESSIAN_INPUT_ACTIVATION, HESSIAN_TRACE, 0),
            (SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE, MEAN_MAX, MEAN_MAX * MAX_BASELINE_SCORE),
            (SensitivityMetric.MEAN_ACTIVATION_VARIANCE, MEAN_VAR, MEAN_VAR * MAX_BASELINE_SCORE),
            (SensitivityMetric.MAX_ACTIVATION_VARIANCE, MAX_VAR, MAX_VAR * MAX_BASELINE_SCORE),
        ),
    )
    def test_data_based_criterion(self, mode, ref_score, ref_act_score, mocker):
        model = self.get_matmul_model()
        data = self.cast_to(self.to_tensor(ACTIVATION), dtype=TensorDataType.float32)
        dataset = Dataset([data])
        criterion_cls = MIXED_PRECISION_CRITERIA.get(mode)
        scores_spy = mocker.spy(criterion_cls, "_calc_sensitivity")
        act_scores_spy = mocker.spy(criterion_cls, "_calc_activation_sensitivity")

        compress_weights(
            model,
            mode=CompressWeightsMode.INT4_ASYM,
            ratio=0.5,
            group_size=1,
            dataset=dataset,
            sensitivity_metric=mode,
            all_layers=True,
        )
        scores = scores_spy.spy_return
        act_scores = act_scores_spy.spy_return
        assert math.isclose(scores[0], ref_score, rel_tol=1e-05, abs_tol=1e-08)
        assert math.isclose(ref_act_score, act_scores, rel_tol=1e-05, abs_tol=1e-08)

    @abstractmethod
    def get_sequential_matmul_model(self): ...

    @abstractmethod
    def to_tensor(): ...

    @abstractmethod
    def check_weights(self, model, ref_ids): ...

    @pytest.mark.parametrize(
        ("mode", "all_layers", "ratio", "ref_ids"),
        (
            (SensitivityMetric.WEIGHT_QUANTIZATION_ERROR, True, 1, [0, 1, 2, 3, 4]),
            (SensitivityMetric.WEIGHT_QUANTIZATION_ERROR, True, 0.8, [0, 3, 4]),
            (SensitivityMetric.WEIGHT_QUANTIZATION_ERROR, True, 0.4, [0]),
            (SensitivityMetric.WEIGHT_QUANTIZATION_ERROR, True, 0.2, []),
            (SensitivityMetric.WEIGHT_QUANTIZATION_ERROR, False, 1, [0, 1, 2, 3]),
            (SensitivityMetric.WEIGHT_QUANTIZATION_ERROR, False, 0.8, [0, 1, 3]),
            (SensitivityMetric.WEIGHT_QUANTIZATION_ERROR, False, 0.4, [0]),
            (SensitivityMetric.WEIGHT_QUANTIZATION_ERROR, False, 0.2, []),
            (SensitivityMetric.HESSIAN_INPUT_ACTIVATION, True, 0.8, [0, 1, 2]),
            (SensitivityMetric.HESSIAN_INPUT_ACTIVATION, False, 0.8, [0, 1, 2]),
            (SensitivityMetric.MEAN_ACTIVATION_VARIANCE, True, 0.8, [0, 1, 2]),
            (SensitivityMetric.MEAN_ACTIVATION_VARIANCE, False, 0.8, [0, 1, 2]),
            (SensitivityMetric.MAX_ACTIVATION_VARIANCE, True, 0.8, [0, 1, 2]),
            (SensitivityMetric.MAX_ACTIVATION_VARIANCE, False, 0.8, [0, 1, 2]),
            (SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE, True, 0.8, [0, 1, 2]),
            (SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE, False, 0.8, [0, 1, 2]),
        ),
    )
    def test_mixed_precision(self, mode, all_layers, ratio, ref_ids):
        model = self.get_sequential_matmul_model()
        first = self.to_tensor(np.ones([1, 4, 4], dtype=np.float32))
        second = self.to_tensor(np.arange(16, dtype=np.float32)).reshape(1, 4, 4)
        dataset = Dataset([first, second])
        compressed_model = compress_weights(
            model,
            mode=CompressWeightsMode.INT4_SYM,
            ratio=ratio,
            group_size=1,
            all_layers=all_layers,
            sensitivity_metric=mode,
            dataset=dataset,
        )
        self.check_weights(compressed_model, ref_ids)
