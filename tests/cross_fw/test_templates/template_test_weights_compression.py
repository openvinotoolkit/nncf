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
import math
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Callable, TypeVar
from unittest.mock import patch

import numpy as np
import pytest

import nncf
import nncf.tensor.functions as fns
from nncf import CompressWeightsMode
from nncf import SensitivityMetric
from nncf import nncf_logger
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.data.dataset import Dataset
from nncf.errors import InvalidGroupSizeError
from nncf.quantization import compress_weights
from nncf.quantization.advanced_parameters import AdvancedAWQParameters as AWQParams
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters as CompressionParams
from nncf.quantization.advanced_parameters import AdvancedGPTQParameters as GPTQParams
from nncf.quantization.algorithms.weight_compression.activation_stats import WCTensorStatistic
from nncf.quantization.algorithms.weight_compression.activation_stats import process_stats
from nncf.quantization.algorithms.weight_compression.algorithm import WeightCompression
from nncf.quantization.algorithms.weight_compression.awq import AWQ
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.mixed_precision import MIXED_PRECISION_CRITERIA
from nncf.quantization.algorithms.weight_compression.scale_estimation import ScaleEstimation
from nncf.quantization.algorithms.weight_compression.weight_lowering import integer_quantize_dequantize_weight
from nncf.scopes import IgnoredScope
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")

NON_ZERO_ROW = [-4, 1, 2]
ACTIVATION = [[NON_ZERO_ROW, [0, 0, 0], [0, 0, 0]]]
MAX_VAR = 3.555555  # np.max(np.var(ACTIVATION, 1))
MEAN_VAR = 1.555555  # np.mean(np.var(ACTIVATION, 1))
MEAN_MAX = 2.333333  # np.mean(np.max(np.abs(ACTIVATION), 1))
HESSIAN_TRACE = (16 + 1 + 4) * 2 / 9  # sum(i*i for i in NON_ZERO_ROW) * 2 / ACTIVATION.size
MAX_BASELINE_SCORE = 1 / 1.1920928955078125e-07

INT4_MODES = (CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM)


def get_relative_error(weight_1: Tensor, weight_2: Tensor, axis: int = 0) -> Tensor:
    diff = (weight_1 - weight_2) ** 2
    return fns.mean(diff, axis=axis) / fns.mean(weight_1**2, axis=axis)


# Spy for AWQ
spy_instance = None


class SpyAWQ(AWQ):
    def __init__(self, *agrs, **kwargs):
        global spy_instance
        super().__init__(*agrs, **kwargs)
        spy_instance = self


class SpyWeightCompressionStatisticsContext:
    def __init__(self, mocker):
        self.mocker = mocker
        self.unique_tensor_collectors = set()
        self.statistic_point_spy = None

    def __enter__(self):
        original_method = StatisticPointsContainer.get_algo_statistics_for_node

        def side_effect(*args, **kwargs):
            results = []
            for tc in original_method(*args, **kwargs):
                results.append(tc)
                self.unique_tensor_collectors.add(tc)
            return results

        self.mocker.patch.object(
            StatisticPointsContainer, "get_algo_statistics_for_node", autospec=True, side_effect=side_effect
        )
        self.statistic_point_spy = self.mocker.spy(WeightCompression, "get_statistic_points")
        return self

    def __exit__(self, *args):
        statistic_points = self.statistic_point_spy.spy_return
        number_tensor_collectors = 0
        for node_statistics_points in statistic_points.values():
            for _statistics_point in node_statistics_points:
                tensor_collectors = _statistics_point.algorithm_to_tensor_collectors.values()
                number_tensor_collectors += len(tensor_collectors)

        assert len(self.unique_tensor_collectors) == number_tensor_collectors


class TemplateWeightCompression(ABC):
    # Test Mixed Precision

    @staticmethod
    @abstractmethod
    def cast_to(x: TTensor, dtype: TensorDataType) -> TTensor:
        """Casts a backend tensor to backend tensor with specified dtype."""

    @staticmethod
    @abstractmethod
    def get_matmul_model() -> TModel:
        """Returns a backend model for test_data_based_criterion."""

    @staticmethod
    @abstractmethod
    def get_RoPE_model() -> TModel:
        """Returns a backend model for test_rope_weight_compression."""

    @staticmethod
    @abstractmethod
    def get_SAM_PE_model() -> TModel:
        """Returns a backend model for test_sam_pe_weight_compression."""

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
        dataset = Dataset([data], self.get_transform_func())
        criterion_cls = MIXED_PRECISION_CRITERIA.get(mode)
        scores_spy = mocker.spy(criterion_cls, "_calc_sensitivity")
        act_scores_spy = mocker.spy(criterion_cls, "_calc_activation_sensitivity")

        with SpyWeightCompressionStatisticsContext(mocker):
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

    @staticmethod
    @abstractmethod
    def get_sequential_matmul_model(transpose_a: bool) -> TModel:
        """Returns a backend model for test_mixed_precision."""

    @staticmethod
    @abstractmethod
    def to_tensor(x: TTensor) -> TTensor:
        """Returns a backend tensor."""

    @staticmethod
    @abstractmethod
    def check_weights(model: TModel, ref_ids: list[int], transpose_a=False) -> None:
        """Checks that only weights with specified ids are compressed in int4 format."""

    @staticmethod
    @abstractmethod
    def get_not_supported_algorithms() -> list[str]:
        """
        Returns a list of not supported weight compression algorithms.
        """

    @staticmethod
    @abstractmethod
    def wrap_model(model, data) -> CompressionParams:
        """
        Returns model wrapped with backend specific graph.
        """

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
    @pytest.mark.parametrize("transpose_a", (False, True))
    def test_mixed_precision(self, mode, all_layers, ratio, ref_ids, transpose_a, transpose_a_supported, mocker):
        if transpose_a and not transpose_a_supported:
            pytest.skip("transpose_a is not supported for the current backend")
        model = self.get_sequential_matmul_model(transpose_a=transpose_a)
        input_shape = (4, 4) if transpose_a else (1, 4, 4)
        first = self.to_tensor(np.ones(input_shape, dtype=np.float32))
        second = self.to_tensor(np.arange(16, dtype=np.float32)).reshape(input_shape)
        dataset = Dataset([first, second], self.get_transform_func())
        compressed_model = compress_weights(
            model,
            mode=CompressWeightsMode.INT4_SYM,
            ratio=ratio,
            group_size=1,
            all_layers=all_layers,
            sensitivity_metric=mode,
            dataset=dataset,
        )
        self.check_weights(compressed_model, ref_ids, transpose_a)

    # Scale Estimation Tests

    @staticmethod
    @abstractmethod
    def get_model_for_test_scale_estimation(transpose_a: bool) -> TModel:
        """
        Returns a backend model for test_scale_estimation.
        """

    @staticmethod
    @abstractmethod
    def get_moe_model_for_test_scale_estimation(transpose_a: bool) -> TModel:
        """
        Returns a backend MoE model for test_scale_estimation with 3D weights.
        """

    @staticmethod
    @abstractmethod
    def get_moe_scale_estimation_ref(check_sampling_activation_stats_flow: bool) -> TTensor:
        """
        :param check_sampling_activation_stats_flow: whether we are checking the flow with sampling when processing
            activation statistics
        Returns the reference output of calculate_quantization_params for MoE model.
        """

    @staticmethod
    @abstractmethod
    def get_scale_estimation_ref(check_sampling_activation_stats_flow: bool) -> TTensor:
        """
        :param check_sampling_activation_stats_flow: whether we are checking the flow with sampling when processing
            activation statistics
        Returns the reference output of calculate_quantization_params of ScaleEstimation.
        """

    @pytest.mark.parametrize("transpose_a", [False, True], ids=["no_tr_a", "tr_a"])
    @pytest.mark.parametrize("is_moe", [False, True], ids=["reg", "moe"])
    @pytest.mark.parametrize("check_sampling_activation_stats_flow", [False, True], ids=["full", "sampled"])
    @pytest.mark.parametrize("is_moe", [False, True])
    @pytest.mark.parametrize("check_sampling_activation_stats_flow", [False, True])
    def test_scale_estimation(
        self, mocker, transpose_a, is_moe, check_sampling_activation_stats_flow, transpose_a_supported
    ):
        """Checks that scales match the reference."""
        if transpose_a and not transpose_a_supported:
            msg = "Transpose a is not supported for the current backend"
            pytest.skip(msg)

        calc_q_params_spy = mocker.spy(ScaleEstimation, "calculate_quantization_params")

        if is_moe:
            model = self.get_moe_model_for_test_scale_estimation(transpose_a=transpose_a)
            input = np.arange(0, 2 * 4 * 8, dtype=np.float32).reshape(2, 4, 8)
        else:
            model = self.get_model_for_test_scale_estimation(transpose_a=transpose_a)
            input = np.arange(0, 4 * 8, dtype=np.float32).reshape(1, 4, 8)

        # prepare dataset of size subset_size with input tensors
        subset_size = 2 if check_sampling_activation_stats_flow else 1
        # make sure that subset size for SE < subset size for statistics collection.
        # This is to test the Optimized statistics processing flow which samples only a few data
        # points in nncf/quantization/algorithms/weight_compression/activation_stats.py
        se_subset_size = subset_size // 2 if check_sampling_activation_stats_flow else subset_size
        input = self.to_tensor(input)

        dataset = Dataset([input + i for i in range(subset_size)], self.get_transform_func())

        with SpyWeightCompressionStatisticsContext(mocker):
            _ = compress_weights(
                model,
                mode=CompressWeightsMode.INT4_ASYM,
                ratio=1.0,
                group_size=8,
                scale_estimation=True,
                all_layers=True,
                dataset=dataset,
                subset_size=subset_size,
                advanced_parameters=nncf.AdvancedCompressionParameters(
                    scale_estimation_params=nncf.AdvancedScaleEstimationParameters(subset_size=se_subset_size)
                ),
            )

        computed_scale = calc_q_params_spy.spy_return[0]

        if is_moe:
            reference = self.get_moe_scale_estimation_ref(check_sampling_activation_stats_flow)
        else:
            reference = self.get_scale_estimation_ref(check_sampling_activation_stats_flow)
        assert fns.allclose(Tensor(reference), computed_scale)

    @staticmethod
    @abstractmethod
    def get_orig_weight(model: TModel) -> Tensor:
        """Returns original weight."""

    @staticmethod
    @abstractmethod
    def get_decompressed_weight(compressed_model: TModel, input: TTensor) -> Tensor:
        """Returns decompressed weight"""

    def test_scale_estimation_outlier_channel_has_lowest_error(self, mocker):
        """Checks that outlier channel has a lowest error after quantization."""
        OUTLIER_CHANNEL = 4
        model = self.get_model_for_test_scale_estimation(transpose_a=False)
        original_weight = self.get_orig_weight(model)

        # prepare dataset with one input tensor
        input = np.arange(0, 4 * 8, dtype=np.float32).reshape(1, 4, 8)
        input[:, :, OUTLIER_CHANNEL] *= 1000  # make one channel relatively higher, should have lowest error.
        input = self.to_tensor(input)
        dataset = Dataset([input], self.get_transform_func())

        with SpyWeightCompressionStatisticsContext(mocker):
            compressed_model = compress_weights(
                model,
                mode=CompressWeightsMode.INT4_ASYM,
                ratio=1.0,
                group_size=-1,
                scale_estimation=True,
                all_layers=True,
                dataset=dataset,
            )

        reduction_axes = self.get_reduction_axes()
        decompressed_weight_before_se = integer_quantize_dequantize_weight(
            original_weight,
            config=WeightCompressionConfig(CompressWeightsMode.INT4_ASYM, -1),
            reduction_axes=reduction_axes,
        )
        decompressed_weight_after_se = self.get_decompressed_weight(compressed_model, input)
        error_before_se = get_relative_error(original_weight, decompressed_weight_before_se, axis=1 - reduction_axes)
        error_after_se = get_relative_error(original_weight, decompressed_weight_after_se, axis=1 - reduction_axes)
        assert fns.argsort(error_after_se)[0] == OUTLIER_CHANNEL  # the smallest error on the outlier channel
        assert error_before_se[OUTLIER_CHANNEL] > error_after_se[OUTLIER_CHANNEL]

    # AWQ Tests
    @staticmethod
    @abstractmethod
    def get_awq_act_model(is_3d_weights, with_multiply, n_layers):
        "Returns a backend model for test_call_max_var_criterion_with_dataset_by_default_awq_act_matmul."

    @staticmethod
    @abstractmethod
    def get_num_multiply_from_awq(model: TModel) -> int:
        "Returns number of Multiply nodes from AWQ."

    @pytest.fixture
    def int4_mode(self, request):
        return None

    @pytest.mark.parametrize("is_3d_weights", [True, False])
    @pytest.mark.parametrize("with_multiply", (True, False))
    def test_call_max_var_criterion_with_dataset_by_default_awq_act_matmul(
        self, int4_mode, with_multiply, is_3d_weights, mocker
    ):
        n_layers = 8
        n_awq_target = n_layers - 1  # first MatMul is always int8
        model = self.get_awq_act_model(is_3d_weights, with_multiply, n_layers)

        dataset = Dataset([self.to_tensor(np.ones([2, 8, 8], dtype=np.float32))], self.get_transform_func())

        with SpyWeightCompressionStatisticsContext(mocker):
            model = compress_weights(model, mode=int4_mode, ratio=1.0, group_size=2, dataset=dataset, awq=True)

        awq_num = self.get_num_multiply_from_awq(model)
        assert awq_num == n_awq_target

    @staticmethod
    @abstractmethod
    def get_awq_model(non_mergable_pattern: bool, is_3d_weights: bool) -> TModel:
        """
        Returns a backend model for test_awq_with_ignored_scope."
        :param is_3d_weights: The model has 3d weights
        """

    @staticmethod
    @abstractmethod
    def get_different_channel_size_model(channel_sizes: list[int]) -> TModel:
        "Returns a backend model with matmuls having different channel sizes."

    @staticmethod
    @abstractmethod
    def get_num_int4_nodes(model: TModel):
        "Returns number of int4 nodes."

    @staticmethod
    @abstractmethod
    def get_num_int4_group_sizes(model: TModel) -> dict[int, int]:
        "Returns number of int4 nodes for each group size."

    @staticmethod
    @abstractmethod
    def get_ignored_scope_name(is_3d_weights) -> str:
        "Returns ignored scope name for test_awq_with_ignored_scope."

    @pytest.mark.parametrize("is_3d_weights", [True, False])
    def test_awq_with_ignored_scope(self, mocker, is_3d_weights):
        model = self.get_awq_model(non_mergable_pattern=False, is_3d_weights=is_3d_weights)
        sz = 8
        n_samples = 10

        input_shape = [2, 8, sz]

        dataset = Dataset(
            [self.to_tensor(np.ones(input_shape, dtype=np.float32)) for i in range(n_samples)],
            self.get_transform_func(),
        )

        with SpyWeightCompressionStatisticsContext(mocker):
            compressed_model = compress_weights(
                model,
                mode=CompressWeightsMode.INT4_SYM,
                ratio=1.0,
                group_size=-1,
                dataset=dataset,
                awq=True,
                ignored_scope=IgnoredScope(names=[self.get_ignored_scope_name(is_3d_weights)]),
            )

        int4_ref_num_compressed = 4  # last MatMul is always int8; one - is ignored; total 6 matmuls
        int4_num_nodes = self.get_num_int4_nodes(compressed_model)
        assert int4_num_nodes == int4_ref_num_compressed, int4_num_nodes

    def test_rope_weight_compression(self):
        model = self.get_RoPE_model()
        sz = 8
        n_samples = 10

        dataset = Dataset(
            [self.to_tensor(np.ones([1, i + 1, sz], dtype=np.float32)) for i in range(n_samples)],
            self.get_transform_func(),
        )
        compressed_model = compress_weights(
            model,
            mode=CompressWeightsMode.INT4_SYM,
            ratio=1.0,
            group_size=-1,
            dataset=dataset,
        )

        int4_ref_num_compressed = 0
        int4_num_nodes = self.get_num_int4_nodes(compressed_model)
        assert int4_num_nodes == int4_ref_num_compressed

    def test_sam_pe_weight_compression(self):
        model = self.get_SAM_PE_model()

        dataset = Dataset(
            [self.to_tensor(np.ones([1, 2, 3, 2], dtype=np.float32))],
            self.get_transform_func(),
        )
        compressed_model = compress_weights(
            model,
            mode=CompressWeightsMode.INT4_SYM,
            ratio=1.0,
            group_size=-1,
            dataset=dataset,
            all_layers=True,
        )

        int4_ref_num_compressed = 0
        int4_num_nodes = self.get_num_int4_nodes(compressed_model)
        assert int4_num_nodes == int4_ref_num_compressed

    @staticmethod
    @abstractmethod
    @pytest.fixture
    def test_awq_scale_ref() -> dict[str, Tensor]:
        "Returns reference for test_awq_scale_reference."

    @abstractmethod
    @pytest.fixture
    def transpose_a_supported(self) -> bool:
        """True if backend supports tranpose for MM activations, False otherwise"""

    # Transpose inputs does not affect mergable pattern code, skippting (True, False)
    @pytest.mark.parametrize("transpose_a,non_mergable_pattern", [(True, True), (False, True), (False, False)])
    @pytest.mark.parametrize("is_3d_weights", [True, False])
    def test_awq_scale_reference(
        self,
        non_mergable_pattern,
        transpose_a,
        test_awq_scale_ref,
        transpose_a_supported,
        is_3d_weights,
        monkeypatch,
        mocker,
    ):
        monkeypatch.setattr("nncf.quantization.algorithms.weight_compression.algorithm.AWQ", SpyAWQ)
        if transpose_a:
            if not transpose_a_supported:
                msg = "Transpose a is not supported for the current backend"
                pytest.skip(msg)

            INPUT_SHAPE = (2, 2, 4) if is_3d_weights else (2, 4)
            model = self.get_transposable_awq_model(
                transpose_a=True, transpose_b=True, input_shape=INPUT_SHAPE, is_3d_weights=is_3d_weights
            )
        else:
            batch_size = 1 if not is_3d_weights else 2
            INPUT_SHAPE = (batch_size, 4, 8)
            model = self.get_awq_model(non_mergable_pattern, is_3d_weights)
        input = 0.01 * np.arange(0, np.multiply.reduce(INPUT_SHAPE), dtype=np.float32).reshape(INPUT_SHAPE) + 0.02
        input = self.to_tensor(input)
        dataset = Dataset([input] * 2, self.get_transform_func())

        with SpyWeightCompressionStatisticsContext(mocker):
            _ = compress_weights(
                model,
                mode=CompressWeightsMode.INT4_SYM,
                ratio=1.0,
                all_layers=transpose_a,
                group_size=-1,
                dataset=dataset,
                awq=True,
            )
        assert spy_instance is not None
        for node_name, scales in spy_instance._scale_per_target_node.items():
            ref = test_awq_scale_ref[is_3d_weights][node_name]
            assert fns.allclose(scales, ref)
            assert scales.shape == ref.shape

    @pytest.mark.parametrize(
        ["group_size", "fallback_mode", "min_adjusted_group_size", "expected_outcome"],
        [
            (32, nncf.GroupSizeFallbackMode.ERROR, None, "exception"),
            (32, nncf.GroupSizeFallbackMode.IGNORE, 16, "warn_ignored"),
            (32, nncf.GroupSizeFallbackMode.ADJUST, 16, "info_cant_adjust"),
            (32, nncf.GroupSizeFallbackMode.ADJUST, 8, "info_adjusted_group_size"),
            (32, None, None, "exception"),
        ],
    )
    def test_error_message_for_invalid_group_size(
        self,
        group_size,
        fallback_mode,
        min_adjusted_group_size,
        expected_outcome,
    ):
        """
        Verifies that:
            - an exception is raised for an invalid group size
            - a warning message is logged when a node is ignored due to an invalid group size
            - an info message is logged when an adjustable group size value cannot be found
            - an info message is logged when the group size is adjusted to a valid value
        """

        model = self.get_different_channel_size_model([8, 8, 8, 8, 8, 8, 8, 16, 32])
        input_example = self.to_tensor(np.ones([1, 8, 8], dtype=np.float32))
        dataset = Dataset([input_example], self.get_transform_func())
        kwargs = dict(
            model=model,
            mode=CompressWeightsMode.INT4_ASYM,
            ratio=0.9,
            group_size=group_size,
            all_layers=True,
            dataset=dataset,
        )
        if fallback_mode is not None or min_adjusted_group_size is not None:
            kwargs["advanced_parameters"] = nncf.AdvancedCompressionParameters(
                group_size_fallback_mode=fallback_mode,
                min_adjusted_group_size=min_adjusted_group_size,
            )

        if expected_outcome == "exception":
            with pytest.raises(InvalidGroupSizeError) as exc_info:
                compress_weights(**kwargs)

            assert "Failed to apply group-wise quantization with group size value" in str(exc_info.value)
        elif expected_outcome == "warn_ignored":
            with patch.object(nncf_logger, "warning") as mock_warning:
                compress_weights(**kwargs)
            warning_messages = [args[0] for args, _ in mock_warning.call_args_list]
            warn_msg = "They will be ignored and kept with original precision."
            assert any(warn_msg in msg for msg in warning_messages)
        elif expected_outcome in ["info_adjusted_group_size", "info_cant_adjust"]:
            with patch.object(nncf_logger, "info") as mock_info:
                compress_weights(**kwargs)
            info_messages = [args[0] for args, _ in mock_info.call_args_list]
            info_msg = (
                "Adjusted group size values will be used"
                if expected_outcome == "info_adjusted_group_size"
                else "A valid adjusted group size value can't be found for some nodes."
            )
            assert any(info_msg in msg for msg in info_messages)
            if expected_outcome == "info_adjusted_group_size":
                table_rows = [
                    "int8_asym, per-channel    │ 50% (1 / 9)                 │ 50% (1 / 9)",
                    "int4_asym, group size 8   │ 25% (7 / 9)                 │ 25% (7 / 9)",
                    "int4_asym, group size 16  │ 25% (1 / 9)                 │ 25% (1 / 9)",
                ]
                for row in table_rows:
                    # On Windows "|" is printed instead of "│"
                    assert any(row in msg.replace("|", "│") for msg in info_messages), "\n".join(info_messages)

    @pytest.mark.parametrize(
        [
            "model_channel_sizes",
            "ratio",
            "group_size",
            "fallback_mode",
            "min_adjusted_group_size",
            "ref_num_group_sizes",
        ],
        [
            ([8, 8, 16, 16, 16, 32], 1.0, 32, nncf.GroupSizeFallbackMode.IGNORE, None, {32: 1}),
            ([8, 8, 16, 16, 16, 32], 1.0, 32, nncf.GroupSizeFallbackMode.ADJUST, 16, {16: 3, 32: 1}),
            ([8, 8, 16, 16, 16, 32], 1.0, 32, nncf.GroupSizeFallbackMode.ADJUST, 32, {32: 1}),
            ([8, 8, 16, 16, 16, 32], 0.5, 32, nncf.GroupSizeFallbackMode.ADJUST, 16, {16: 2}),
        ],
    )
    def test_group_size_fallback_modes(
        self,
        model_channel_sizes,
        ratio,
        group_size,
        fallback_mode,
        min_adjusted_group_size,
        ref_num_group_sizes,
    ):
        model = self.get_different_channel_size_model(model_channel_sizes)
        input_example = self.to_tensor(np.ones([1, model_channel_sizes[0], model_channel_sizes[0]], dtype=np.float32))
        dataset = Dataset([input_example], self.get_transform_func())
        kwargs = dict(
            model=model,
            mode=CompressWeightsMode.INT4_SYM,
            ratio=ratio,
            all_layers=True,
            group_size=group_size,
            dataset=dataset,
        )
        if fallback_mode is not None:
            kwargs["advanced_parameters"] = nncf.AdvancedCompressionParameters(
                group_size_fallback_mode=fallback_mode,
                min_adjusted_group_size=min_adjusted_group_size,
            )

        model = compress_weights(**kwargs)

        num_group_sizes = self.get_num_int4_group_sizes(model)
        assert ref_num_group_sizes == num_group_sizes, (
            f"Expected {ref_num_group_sizes} group size values, but got {num_group_sizes}."
        )

    @pytest.mark.parametrize("is_3d_weights", [True, False])
    @pytest.mark.parametrize("dataset", [None, np.ones([2, 8, 8], dtype=np.float32)])
    @pytest.mark.parametrize("prefer_data_aware_scaling", [True, False])
    def test_data_free_awq(self, dataset, prefer_data_aware_scaling, is_3d_weights, mocker):
        input_data = np.ones([2, 8, 8], dtype=np.float32)

        n_layers = 8
        n_awq_target = n_layers - 1  # first MatMul is always int8
        model = self.get_awq_act_model(is_3d_weights, True, n_layers)
        model = self.wrap_model(model, input_data)

        if dataset is not None:
            dataset = Dataset([self.to_tensor(dataset)], self.get_transform_func())

        fn_name = "_data_free_step" if dataset is None or not prefer_data_aware_scaling else "_data_aware_step"

        collect_spy = mocker.spy(AWQ, fn_name)

        compressed_model = compress_weights(
            model,
            mode=CompressWeightsMode.INT4_ASYM,
            ratio=1.0,
            group_size=-1,
            dataset=dataset,
            awq=True,
            advanced_parameters=CompressionParams(
                awq_params=AWQParams(
                    prefer_data_aware_scaling=prefer_data_aware_scaling,
                )
            ),
        )

        n_awq = self.get_num_multiply_from_awq(compressed_model)
        assert n_awq == n_awq_target
        assert collect_spy.call_count == n_awq, f"Statistics should be collected {n_awq_target} times."

    @staticmethod
    def get_transform_func() -> Callable[..., Any] | None:
        return None

    @staticmethod
    def get_reduction_axes() -> int:
        return 1

    @dataclass
    class ProcessStatsTestCase:
        reduced_shape: tuple[int, ...]
        activation_shapes: list[tuple[int, ...]]
        subset_size: int
        ref_s: np.ndarray
        ref_X: np.ndarray
        act_ch_axis: int | None = None

    @pytest.mark.parametrize(
        "case",
        [
            # 2D Activations
            ProcessStatsTestCase(
                reduced_shape=(2,),
                activation_shapes=[(1, 2), (3, 2), (5, 2), (10, 2)],
                subset_size=2,
                ref_s=np.array([6, 7]),
                ref_X=np.array([6, 2, 7, 3]).reshape(2, 2),
            ),
            ProcessStatsTestCase(
                reduced_shape=(2,),
                activation_shapes=[(2, 1), (2, 3), (2, 5), (2, 10)],
                subset_size=2,
                act_ch_axis=0,
                ref_s=np.array([6, 7]),
                ref_X=np.array([6, 2, 7, 3]).reshape(2, 2),
            ),
            ProcessStatsTestCase(
                reduced_shape=(2,),
                activation_shapes=[(5, 2), (5, 2)],
                subset_size=2,
                ref_s=np.array([2, 3]),
                ref_X=np.array([0, 2, 1, 3]).reshape(2, 2),
            ),
            # 3D Activations
            ProcessStatsTestCase(
                reduced_shape=(2, 4),
                activation_shapes=[(1, 2, 4), (3, 2, 4), (5, 2, 4), (10, 2, 4)],
                subset_size=2,
                ref_s=np.array(list(range(24, 32))).reshape(2, 4),
                ref_X=np.array([24, 8, 25, 9, 26, 10, 27, 11, 28, 12, 29, 13, 30, 14, 31, 15]).reshape(2, 4, 2),
            ),
            ProcessStatsTestCase(
                reduced_shape=(2, 4),
                activation_shapes=[(1, 100000, 2, 4), (3, 10000, 2, 4), (5, 1000, 2, 4), (10, 5, 2, 4)],
                subset_size=2,
                act_ch_axis=1,
                ref_s=np.array(list(range(24, 32))).reshape(2, 4),
                ref_X=np.array([24, 8, 25, 9, 26, 10, 27, 11, 28, 12, 29, 13, 30, 14, 31, 15]).reshape(2, 4, 2),
            ),
            ProcessStatsTestCase(
                reduced_shape=(2, 4),
                activation_shapes=[(1, 2, 4), (1, 2, 4)],
                subset_size=2,
                ref_s=np.array(list(range(8, 16))).reshape(2, 4),
                ref_X=np.array([0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]).reshape(2, 4, 2),
            ),
        ],
    )
    def test_process_stats(self, case: ProcessStatsTestCase):
        total_elements = reduce(mul, case.reduced_shape, 1)
        mean_values = [
            Tensor(
                np.arange(i * total_elements, (i + 1) * total_elements, dtype=np.float32).reshape(case.reduced_shape)
            )
            for i in range(len(case.activation_shapes))
        ]

        stats = WCTensorStatistic(mean_values=mean_values, shape_values=case.activation_shapes)

        if case.act_ch_axis is None:
            s, X = process_stats(stats, case.subset_size)
        else:
            s, X = process_stats(stats, case.subset_size, case.act_ch_axis)

        assert s.shape == case.ref_s.shape
        assert fns.allclose(s, self.to_tensor(case.ref_s))
        assert X.shape == case.ref_X.shape
        assert fns.allclose(X, self.to_tensor(case.ref_X))

    @staticmethod
    @abstractmethod
    def get_transposable_awq_model(
        transpose_a: bool, transpose_b: bool, input_shape=None, is_3d_weights: bool = False
    ) -> TModel:
        "Returns a backend model for test_compression_with_transpose."

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(lora_correction=True),
            dict(
                gptq=True,
                advanced_parameters=CompressionParams(gptq_params=GPTQParams(subset_size=2)),
            ),
        ],
    )
    def test_compression_skipped_with_transposed_activations(self, transpose_a_supported, kwargs):
        if not transpose_a_supported:
            pytest.skip("transpose_a is not supported for the current backend")
        if kwargs.get("gptq", False) and "gptq" in self.get_not_supported_algorithms():
            pytest.skip("GPTQ is not supported")
        if kwargs.get("lora_correction", False) and "lora_correction" in self.get_not_supported_algorithms():
            pytest.skip("lora_correction is not supported")

        INPUT_SHAPE = (2, 4)
        model = self.get_transposable_awq_model(transpose_a=True, transpose_b=True, input_shape=INPUT_SHAPE)
        input = 0.01 * np.arange(0, np.multiply.reduce(INPUT_SHAPE), dtype=np.float32).reshape(INPUT_SHAPE) + 0.02
        input = self.to_tensor(input)
        dataset = Dataset([input] * 2, self.get_transform_func())

        with pytest.raises(nncf.UnsupportedModelError):
            compress_weights(
                model,
                mode=CompressWeightsMode.INT4_SYM,
                ratio=1.0,
                group_size=1,
                subset_size=2,
                dataset=dataset,
                all_layers=True,
                **kwargs,
            )
