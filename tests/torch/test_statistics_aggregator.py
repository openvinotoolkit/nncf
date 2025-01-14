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

from typing import Callable, List, Type

import numpy as np
import pytest
import torch
from torch import nn

from nncf import Dataset
from nncf.common import factory
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.experimental.common.tensor_statistics.collectors import TensorReducerBase
from nncf.quantization.algorithms.fast_bias_correction.torch_backend import PTFastBiasCorrectionAlgoBackend
from nncf.quantization.algorithms.min_max.torch_backend import PTMinMaxAlgoBackend
from nncf.quantization.range_estimator import RangeEstimatorParametersSet
from nncf.torch.dynamic_graph.patch_pytorch import register_operator
from nncf.torch.graph.graph import PTTargetPoint
from nncf.torch.model_transformer import PTInsertionCommand
from nncf.torch.statistics.aggregator import PTStatisticsAggregator
from tests.common.test_statistics_aggregator import TemplateTestStatisticsAggregator
from tests.torch.helpers import HookChecker
from tests.torch.ptq.helpers import get_nncf_network
from tests.torch.ptq.test_ptq_params import ToNNCFNetworkInterface

IDENTITY_NODE_NAME = "PTIdentityConvModel/__add___0"
CONV_NODE_NAME = "PTIdentityConvModel/Conv2d[conv]/conv2d_0"
INPUT_SHAPE = [1, 3, 3, 3]


class PTIdentityConvModel(nn.Module, ToNNCFNetworkInterface):
    def __init__(self, kernel):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3)
        self.conv.weight.data = torch.tensor(kernel, dtype=torch.float32)

    def forward(self, x):
        return self.conv(x + 0.0)

    def get_nncf_network(self):
        return get_nncf_network(self, INPUT_SHAPE)


MinMaxTestParameters = TemplateTestStatisticsAggregator.MinMaxTestParameters


class TestStatisticsAggregator(TemplateTestStatisticsAggregator):
    @staticmethod
    def get_min_max_algo_backend_cls() -> Type[PTMinMaxAlgoBackend]:
        return PTMinMaxAlgoBackend

    def get_bias_correction_algo_backend_cls(self) -> None:
        pytest.skip("PTBiasCorrectionAlgoBackend is not implemented")

    def get_fast_bias_correction_algo_backend_cls(self) -> Type[PTFastBiasCorrectionAlgoBackend]:
        return PTFastBiasCorrectionAlgoBackend

    def get_backend_model(self, dataset_samples):
        sample = dataset_samples[0].reshape(INPUT_SHAPE[1:])
        conv_w = self.dataset_samples_to_conv_w(np.array(sample))
        return PTIdentityConvModel(conv_w).get_nncf_network()

    @pytest.fixture
    def is_backend_support_custom_estimators(self) -> bool:
        return True

    @pytest.fixture(scope="session")
    def test_params(self):
        return

    def get_statistics_aggregator(self, dataset):
        return PTStatisticsAggregator(dataset)

    def get_dataset(self, samples):
        def transform_fn(data_item):
            return data_item

        return Dataset(samples, transform_fn)

    @staticmethod
    def get_target_point(target_type: TargetType):
        target_node_name = IDENTITY_NODE_NAME
        port_id = 0
        if target_type == TargetType.OPERATION_WITH_WEIGHTS:
            target_node_name = CONV_NODE_NAME
            port_id = 1
        return PTMinMaxAlgoBackend.target_point(target_type, target_node_name, port_id)

    def get_target_point_cls(self):
        return PTTargetPoint

    def reducers_map(self) -> List[TensorReducerBase]:
        return None

    @pytest.fixture
    def dataset_samples(self, dataset_values):
        input_shape = INPUT_SHAPE
        dataset_samples = [np.zeros(input_shape), np.ones(input_shape)]

        for i, value in enumerate(dataset_values):
            dataset_samples[0][0, i, 0, 0] = value["max"]
            dataset_samples[0][0, i, 0, 1] = value["min"]

        return torch.tensor(dataset_samples, dtype=torch.float32)

    @pytest.fixture(params=[False], ids=["out_of_palce"])
    def inplace_statistics(self, request) -> bool:
        return request.param

    @pytest.mark.skip("Merging is not implemented yet")
    def test_statistics_merging_simple(self, dataset_samples, inplace_statistics, statistic_point_params):
        pass

    @pytest.mark.skip("Merging is not implemented yet")
    def test_statistic_merging(self, dataset_samples, inplace_statistics):
        pass

    @pytest.mark.skip("Merging is not implemented yet")
    def test_same_collectors_different_attrs_dont_merge(self, statistics_type, test_params, dataset_samples):
        pass

    @pytest.mark.parametrize(
        "test_parameters",
        (
            MinMaxTestParameters(
                RangeEstimatorParametersSet.MINMAX,
                TargetType.OPERATOR_PRE_HOOK,
                QuantizationMode.SYMMETRIC,
                False,
                256,
                -256,
            ),
            MinMaxTestParameters(
                RangeEstimatorParametersSet.MINMAX,
                TargetType.OPERATOR_POST_HOOK,
                QuantizationMode.SYMMETRIC,
                False,
                256,
                -256,
            ),
        ),
    )
    def test_successive_statistics_aggregation(
        self,
        test_parameters: MinMaxTestParameters,
        dataset_samples,
        inplace_statistics,
        is_backend_support_custom_estimators,
        mocker,
    ):
        is_stat_in_shape_of_scale = True
        model = self.get_backend_model(dataset_samples)
        quantizer_config = QuantizerConfig(
            mode=test_parameters.quantization_mode, per_channel=test_parameters.per_channel
        )

        is_standard_estimator = test_parameters.range_estimator_params in [
            RangeEstimatorParametersSet.MINMAX,
            RangeEstimatorParametersSet.MEAN_MINMAX,
        ]
        if not is_standard_estimator and not is_backend_support_custom_estimators:
            pytest.skip("Custom estimators are not supported for this backend yet")

        # Register operations before statistic collection
        def fn(x):
            return x * 2

        target_point = self.get_target_point(test_parameters.target_type)
        model = self.__add_fn_to_model(model, target_point, fn)

        # Check hook inserted correctly
        self.__check_successive_hooks(model, target_point, fn)

        # Register and collect statistics after inserted operations
        statistic_points = self.__get_statistic_points(
            test_parameters, model, quantizer_config, dataset_samples, inplace_statistics, mocker
        )
        tensor_collector = self.__collect_statistics_get_collector(statistic_points, model, dataset_samples)
        # Check values are changed because of the inserted operation
        self.__check_collector(
            test_parameters,
            tensor_collector,
            is_stat_in_shape_of_scale,
        )

        # Check the inserted operation is inside the model
        self.__check_successive_hooks(model, target_point, fn)

    @pytest.mark.parametrize(
        "test_parameters, nested_target_node_name",
        (
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MINMAX,
                    TargetType.OPERATOR_PRE_HOOK,
                    QuantizationMode.SYMMETRIC,
                    False,
                    512,
                    -512,
                ),
                "PTIdentityConvModel/fn_0",
            ),
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MINMAX,
                    TargetType.OPERATOR_POST_HOOK,
                    QuantizationMode.SYMMETRIC,
                    False,
                    512,
                    -512,
                ),
                "PTIdentityConvModel/fn_0",
            ),
        ),
    )
    @pytest.mark.parametrize("nested_target_type", [TargetType.OPERATOR_PRE_HOOK, TargetType.OPERATOR_POST_HOOK])
    def test_nested_statistics_aggregation(
        self,
        test_parameters: MinMaxTestParameters,
        nested_target_type: TargetType,
        nested_target_node_name,
        dataset_samples,
        inplace_statistics,
        is_backend_support_custom_estimators,
        mocker,
    ):
        is_stat_in_shape_of_scale = True
        model = self.get_backend_model(dataset_samples)
        quantizer_config = QuantizerConfig(
            mode=test_parameters.quantization_mode, per_channel=test_parameters.per_channel
        )

        is_standard_estimator = test_parameters.range_estimator_params in [
            RangeEstimatorParametersSet.MINMAX,
            RangeEstimatorParametersSet.MEAN_MINMAX,
        ]
        if not is_standard_estimator and not is_backend_support_custom_estimators:
            pytest.skip("Custom estimators are not supported for this backend yet")

        # Register operations before statistic collection
        @register_operator()
        def fn(x):
            return x * 2

        target_point = self.get_target_point(test_parameters.target_type)
        model = self.__add_fn_to_model(model, target_point, fn)
        nested_target_point = PTMinMaxAlgoBackend.target_point(nested_target_type, nested_target_node_name, 0)
        model = self.__add_fn_to_model(model, nested_target_point, fn)

        # Check hook inserted correctly
        self.__check_nested_hooks(model, target_point, nested_target_type, nested_target_node_name, fn)

        # Register and collect statistics after inserted operations
        statistic_points = self.__get_statistic_points(
            test_parameters, model, quantizer_config, dataset_samples, inplace_statistics, mocker
        )
        tensor_collector = self.__collect_statistics_get_collector(statistic_points, model, dataset_samples)
        # Check values are changed because of the inserted operation
        self.__check_collector(
            test_parameters,
            tensor_collector,
            is_stat_in_shape_of_scale,
        )

        # Check the inserted operation is inside the model
        self.__check_nested_hooks(model, target_point, nested_target_type, nested_target_node_name, fn)

    @staticmethod
    def __add_fn_to_model(model, target_point, fn):
        layout = TransformationLayout()
        command = PTInsertionCommand(target_point, fn)
        layout.register(command)
        model_transformer = factory.ModelTransformerFactory.create(model)
        model = model_transformer.transform(layout)
        model.nncf.rebuild_graph()
        return model

    @classmethod
    def __get_statistic_points(
        cls, test_parameters: MinMaxTestParameters, model, quantizer_config, dataset_samples, inplace_statistics, mocker
    ) -> StatisticPointsContainer:
        statistics_points = StatisticPointsContainer()
        for target_type in [test_parameters.target_type]:
            target_point = cls.get_target_point(target_type)
            statistic_point = cls.create_statistics_point(
                model,
                quantizer_config,
                target_point,
                len(dataset_samples),
                "TEST_ALGO",
                inplace_statistics,
                test_parameters.range_estimator_params,
                mocker,
            )
            statistics_points.add_statistic_point(statistic_point)
        return statistics_points

    def __collect_statistics_get_collector(
        self,
        statistics_points: StatisticPointsContainer,
        model,
        dataset_samples,
    ):
        dataset = self.get_dataset(dataset_samples)
        statistics_aggregator = self.get_statistics_aggregator(dataset)
        statistics_aggregator.register_statistic_points(statistics_points)
        graph = NNCFGraphFactory.create(model)
        statistics_aggregator.collect_statistics(model, graph)

        tensor_collectors = list(statistics_points.get_tensor_collectors())
        assert len(tensor_collectors) == 1
        return tensor_collectors[0][2]

    @staticmethod
    def __check_collector(test_parameters, tensor_collector, stat_in_shape_of_scale):
        stat = tensor_collector.get_statistics()
        # Torch and Openvino backends tensor collectors return values in shape of scale
        # in comparison to ONNX backends.
        ref_min_val, ref_max_val = test_parameters.ref_min_val, test_parameters.ref_max_val
        if isinstance(ref_min_val, np.ndarray) and stat_in_shape_of_scale:
            shape = (1, 3, 1, 1)
            if test_parameters.target_type == TargetType.OPERATION_WITH_WEIGHTS:
                shape = (3, 1, 1, 1)
            ref_min_val, ref_max_val = map(lambda x: np.reshape(x, shape), (ref_min_val, ref_max_val))

        assert np.allclose(stat.min_values.data, ref_min_val)
        assert np.allclose(stat.max_values.data, ref_max_val)
        if isinstance(ref_min_val, np.ndarray):
            assert stat.min_values.shape == ref_min_val.shape
            assert stat.max_values.shape == ref_max_val.shape
        else:
            ref_shape = (1, 1, 1, 1) if stat_in_shape_of_scale else ()
            assert stat.min_values.shape == ref_shape
            assert stat.max_values.shape == ref_shape

    @staticmethod
    def __check_successive_hooks(model: nn.Module, target_point: PTTargetPoint, fn: Callable):
        checker = HookChecker(model, "conv")
        checker.add_ref(
            ref_hooks=[fn],
            target_type=(
                TargetType.OPERATOR_PRE_HOOK
                if target_point.target_type == TargetType.OPERATION_WITH_WEIGHTS
                else target_point.target_type
            ),
            target_node_name=target_point.target_node_name,
            input_port_id=target_point.input_port_id,
        )
        checker.check_with_reference()

    @staticmethod
    def __check_nested_hooks(
        model: nn.Module,
        target_point: PTTargetPoint,
        nested_target_type: TargetType,
        nested_target_node_name: str,
        fn: Callable,
    ):
        checker = HookChecker(model, "conv")
        checker.add_ref(
            ref_hooks=[fn],
            target_type=(
                TargetType.OPERATOR_PRE_HOOK
                if target_point.target_type == TargetType.OPERATION_WITH_WEIGHTS
                else target_point.target_type
            ),
            target_node_name=target_point.target_node_name,
            input_port_id=target_point.input_port_id,
        )
        checker.add_ref(
            ref_hooks=[fn],
            target_type=nested_target_type,
            target_node_name=nested_target_node_name,
            input_port_id=0,
        )
        checker.check_with_reference()
