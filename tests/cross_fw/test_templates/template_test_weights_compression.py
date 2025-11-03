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
from typing import Any, Callable, Optional, TypeVar
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
    def get_sequential_matmul_model() -> TModel:
        """Returns a backend model for test_mixed_precision."""

    @staticmethod
    @abstractmethod
    def to_tensor(x: TTensor) -> TTensor:
        """Returns a backend tensor."""

    @staticmethod
    @abstractmethod
    def check_weights(model: TModel, ref_ids: list[int]) -> None:
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
    def test_mixed_precision(self, mode, all_layers, ratio, ref_ids, mocker):
        model = self.get_sequential_matmul_model()
        first = self.to_tensor(np.ones([1, 4, 4], dtype=np.float32))
        second = self.to_tensor(np.arange(16, dtype=np.float32)).reshape(1, 4, 4)
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
        self.check_weights(compressed_model, ref_ids)

    # Scale Estimation Tests

    @staticmethod
    @abstractmethod
    def get_model_for_test_scale_estimation() -> TModel:
        """
        Returns a backend model for test_scale_estimation.
        """

    @staticmethod
    @abstractmethod
    def get_moe_model_for_test_scale_estimation() -> TModel:
        """
        Returns a backend MoE model for test_scale_estimation with 3D weights.
        """

    @staticmethod
    @abstractmethod
    def get_moe_scale_estimation_ref() -> TTensor:
        """
        Returns the reference output of calculate_quantization_params for MoE model.
        """

    @staticmethod
    @abstractmethod
    def get_scale_estimation_ref() -> TTensor:
        """
        Returns the reference output of calculate_quantization_params of ScaleEstimation.
        """

    @pytest.mark.parametrize("is_moe", [False, True])
    def test_scale_estimation(self, mocker, is_moe):
        """Checks that scales match the reference."""
        calc_q_params_spy = mocker.spy(ScaleEstimation, "calculate_quantization_params")

        if is_moe:
            model = self.get_moe_model_for_test_scale_estimation()
            input = np.arange(0, 4 * 4 * 8, dtype=np.float32).reshape(4, 4, 8)
        else:
            model = self.get_model_for_test_scale_estimation()
            input = np.arange(0, 4 * 8, dtype=np.float32).reshape(1, 4, 8)

        # prepare dataset with one input tensor
        input = self.to_tensor(input)
        dataset = Dataset([input], self.get_transform_func())

        with SpyWeightCompressionStatisticsContext(mocker):
            _ = compress_weights(
                model,
                mode=CompressWeightsMode.INT4_ASYM,
                ratio=1.0,
                group_size=8,
                scale_estimation=True,
                all_layers=True,
                dataset=dataset,
            )

        computed_scale = calc_q_params_spy.spy_return[0]

        if is_moe:
            reference = self.get_moe_scale_estimation_ref()
        else:
            reference = self.get_scale_estimation_ref()

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
        model = self.get_model_for_test_scale_estimation()
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
    def get_awq_act_model(with_multiply, n_layers):
        "Returns a backend model for test_call_max_var_criterion_with_dataset_by_default_awq_act_matmul."

    @staticmethod
    @abstractmethod
    def get_num_multiply_from_awq(model: TModel) -> int:
        "Returns number of Multiply nodes from AWQ."

    @pytest.fixture
    def int4_mode(self, request):
        return None

    @pytest.mark.parametrize("with_multiply", (True, False))
    def test_call_max_var_criterion_with_dataset_by_default_awq_act_matmul(self, int4_mode, with_multiply, mocker):
        n_layers = 8
        n_awq_target = n_layers - 1  # first MatMul is always int8
        model = self.get_awq_act_model(with_multiply, n_layers)

        dataset = Dataset([self.to_tensor(np.ones([1, 8, 8], dtype=np.float32))], self.get_transform_func())

        with SpyWeightCompressionStatisticsContext(mocker):
            model = compress_weights(model, mode=int4_mode, ratio=1.0, group_size=2, dataset=dataset, awq=True)

        awq_num = self.get_num_multiply_from_awq(model)
        assert awq_num == n_awq_target

    @staticmethod
    @abstractmethod
    def get_awq_model() -> TModel:
        "Returns a backend model for test_awq_with_ignored_scope."

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
    def get_ignored_scope_name() -> str:
        "Returns ignored scope name for test_awq_with_ignored_scope."

    def test_awq_with_ignored_scope(self, mocker):
        model = self.get_awq_model()
        sz = 8
        n_samples = 10

        dataset = Dataset(
            [self.to_tensor(np.ones([1, 8, sz], dtype=np.float32)) for i in range(n_samples)],
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
                ignored_scope=IgnoredScope(names=[self.get_ignored_scope_name()]),
            )

        int4_ref_num_compressed = 4  # last MatMul is always int8; one - is ignored; total 6 matmuls
        int4_num_nodes = self.get_num_int4_nodes(compressed_model)
        assert int4_num_nodes == int4_ref_num_compressed

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

    @staticmethod
    @abstractmethod
    def get_reference_for_test_awq_scale_reference() -> dict[str, Tensor]:
        "Returns reference for test_awq_scale_reference."

    def test_awq_scale_reference(self, monkeypatch, mocker):
        monkeypatch.setattr("nncf.quantization.algorithms.weight_compression.algorithm.AWQ", SpyAWQ)
        model = self.get_awq_model()

        input = 0.01 * np.arange(0, 4 * 8, dtype=np.float32).reshape(1, 4, 8) + 0.02
        input = self.to_tensor(input)
        dataset = Dataset([input], self.get_transform_func())

        with SpyWeightCompressionStatisticsContext(mocker):
            _ = compress_weights(
                model,
                mode=CompressWeightsMode.INT4_SYM,
                ratio=1.0,
                group_size=-1,
                dataset=dataset,
                awq=True,
            )
        assert spy_instance is not None
        for node_name, scales in spy_instance._scale_per_target_node.items():
            assert fns.allclose(scales, self.get_reference_for_test_awq_scale_reference()[node_name])

    @pytest.mark.parametrize("algorithm", (None, "awq", "scale_estimation", "gptq", "lora_correction"))
    @pytest.mark.parametrize(
        ["group_size", "fallback_mode", "min_adjusted_group_size", "expected_outcome"],
        [
            (32, nncf.GroupSizeFallbackMode.ERROR, None, "exception"),
            (32, nncf.GroupSizeFallbackMode.IGNORE, 16, "warn_ignored"),
            (32, nncf.GroupSizeFallbackMode.ADJUST, 16, "info_cant_adjust"),
            (32, nncf.GroupSizeFallbackMode.ADJUST, 8, "info_adjusted_group_size"),
            (32, None, None, "warn_ignored"),
        ],
    )
    def test_error_message_for_invalid_group_size(
        self,
        algorithm,
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
        if algorithm in self.get_not_supported_algorithms():
            pytest.skip("Skipping test for not supported algorithms")

        model = self.get_awq_model()
        hidden_dim = 8
        input_example = self.to_tensor(np.ones([1, 4, hidden_dim], dtype=np.float32))
        dataset = Dataset([input_example], self.get_transform_func())
        algorithm_dict = {algorithm: True} if algorithm else {}
        kwargs = dict(
            model=model,
            mode=CompressWeightsMode.INT4_ASYM,
            ratio=1.0,
            group_size=group_size,
            all_layers=True,
            **algorithm_dict,
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
                "Adjusted group size values will be used:"
                if expected_outcome == "info_adjusted_group_size"
                else "A valid adjusted group size value can't be found for some nodes."
            )
            assert any(info_msg in msg for msg in info_messages)

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
            ([8, 8, 16, 16, 16, 32], 1.0, 32, None, None, {32: 1}),
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

    @pytest.mark.parametrize("dataset", [None, np.ones([1, 8, 8], dtype=np.float32)])
    @pytest.mark.parametrize("prefer_data_aware_scaling", [True, False])
    def test_data_free_awq(self, dataset, prefer_data_aware_scaling, mocker):
        input_data = np.ones([1, 8, 8], dtype=np.float32)

        n_layers = 8
        n_awq_target = n_layers - 1  # first MatMul is always int8
        model = self.get_awq_act_model(True, n_layers)
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
    def get_transform_func() -> Optional[Callable[..., Any]]:
        return None

    @staticmethod
    def get_reduction_axes() -> int:
        return 1
