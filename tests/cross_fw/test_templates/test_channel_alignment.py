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

from abc import abstractmethod
from typing import Type

import numpy as np
import pytest

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationType
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.experimental.common.tensor_statistics.collectors import MedianAggregator
from nncf.experimental.common.tensor_statistics.collectors import QuantileReducer
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.quantization.algorithms.channel_alignment.algorithm import ChannelAlignment
from nncf.quantization.algorithms.channel_alignment.backend import ChannelAlignmentAlgoBackend
from nncf.quantization.algorithms.channel_alignment.backend import LayoutDescriptor
from tests.cross_fw.test_templates.models import NNCFGraphCA
from tests.cross_fw.test_templates.models import NNCFGraphCAWithBias

EPS = 1e-3

VALID_CONV_LAYER_ATTR = ConvolutionLayerAttributes(
    weight_requires_grad=False,
    in_channels=5,
    out_channels=5,
    kernel_size=(5, 5),
    stride=(1, 1),
    dilations=(1, 1),
    groups=1,
    transpose=False,
    padding_values=(0, 0, 0, 0),
)


DEPTHWISE_CONV_LAYER_ATTR = ConvolutionLayerAttributes(
    weight_requires_grad=False,
    in_channels=5,
    out_channels=1,
    kernel_size=(5, 5),
    stride=(1, 1),
    dilations=(1, 1),
    groups=5,
    transpose=False,
    padding_values=(0, 0, 0, 0),
)

MATMUL_LAYER_METATYPES = [
    # 2D
    LinearLayerAttributes(
        weight_requires_grad=False,
        in_features=5,
        out_features=10,
        with_bias=False,
    ),
    # 1D
    LinearLayerAttributes(
        weight_requires_grad=False,
        in_features=5,
        out_features=None,
        with_bias=False,
    ),
    # 5D
    LinearLayerAttributes(
        weight_requires_grad=False,
        in_features=5,
        out_features=None,
        with_bias=False,
    ),
]


INVALID_CONSUMER_CONV_LAYER_ATTRS = [
    ConvolutionLayerAttributes(
        weight_requires_grad=False,
        in_channels=5,
        out_channels=5,
        kernel_size=(5, 5),
        stride=(2, 1),
        dilations=(1, 1),
        groups=1,
        transpose=False,
        padding_values=(0, 0, 0, 0),
    ),
    ConvolutionLayerAttributes(
        weight_requires_grad=False,
        in_channels=5,
        out_channels=5,
        kernel_size=(5, 5),
        stride=(1, 1),
        dilations=(2, 1),
        groups=1,
        transpose=False,
        padding_values=(0, 0, 0, 0),
    ),
    ConvolutionLayerAttributes(
        weight_requires_grad=False,
        in_channels=5,
        out_channels=5,
        kernel_size=(5, 5),
        stride=(1, 1),
        dilations=(2, 1),
        groups=1,
        transpose=False,
        padding_values=(0, 0, 0, 0),
    ),
    ConvolutionLayerAttributes(
        weight_requires_grad=False,
        in_channels=5,
        out_channels=5,
        kernel_size=(5, 5),
        stride=(1, 1),
        dilations=(1, 1),
        groups=1,
        transpose=False,
        padding_values=(1, 0, 0, 0),
    ),
]


INVALID_CONV_LAYER_ATTR = ConvolutionLayerAttributes(
    weight_requires_grad=False,
    in_channels=5,
    out_channels=5,
    kernel_size=(5, 5),
    stride=(1, 1),
    dilations=(1, 1),
    groups=5,
    transpose=False,
    padding_values=(0, 0, 0, 0),
)


class TemplateTestChannelAlignment:
    @abstractmethod
    def get_backend_cls(self) -> Type[ChannelAlignmentAlgoBackend]:
        pass

    @abstractmethod
    def target_point(self, target_type: TargetType, target_node_name: str, port_id: int):
        pass

    @abstractmethod
    def convert_conv_layer_attrs(self, layer_attributes):
        pass

    @abstractmethod
    def get_conv_metatype(self):
        pass

    @abstractmethod
    def get_add_metatype(self):
        pass

    @abstractmethod
    def get_add_layer_attrs(self):
        pass

    @abstractmethod
    def get_constant_metatype(self):
        pass

    @abstractmethod
    def get_transformation_commands(self):
        pass

    @abstractmethod
    def mock_command_creation_factory(self, mocker) -> None:
        pass

    def mock_nncf_graph_factory(self, mocker, nncf_graph: NNCFGraph) -> None:
        mocker.patch("nncf.common.factory.NNCFGraphFactory.create", return_value=nncf_graph)

    def mock_model_transformer_factory(self, mocker, model_transformer: ModelTransformer) -> None:
        mocker.patch("nncf.common.factory.ModelTransformerFactory.create", return_value=model_transformer)

    @pytest.mark.parametrize(
        "conv_out_value,refs",
        [
            (np.arange(12).reshape(4, 3), ([-8, -16, -24], [83, 265, 449, 631])),
            (np.arange(24).reshape(4, 3, 2, 1), ([-8, -16, -24], [383, 1105, 1829, 2551])),
        ],
    )
    @pytest.mark.parametrize("transposed", [False, True])
    def test_align_means(self, conv_out_value, refs, transposed):
        amean = np.array([10, 20, 30])
        dims_descriptor = LayoutDescriptor(0, 1, 1)
        if transposed:
            if conv_out_value.ndim == 2:
                conv_out_value = np.transpose(conv_out_value, (1, 0))
                dims_descriptor = LayoutDescriptor(1, 0, 1)
            else:
                conv_out_value = np.transpose(conv_out_value, (3, 1, 2, 0))
                dims_descriptor = LayoutDescriptor(3, 1, 1)
        bias_in_value = np.array([2, 4, 6])
        bias_out_value = np.array([3, 5, 9, 11])
        updated_add_in_vals, updated_add_out_vals = ChannelAlignment._align_means(
            bias_in_value, bias_out_value, conv_out_value, amean, dims_descriptor
        )
        assert np.allclose(updated_add_in_vals, np.array(refs[0]))
        assert np.allclose(updated_add_out_vals, np.array(refs[1]))

    REF_UPDATED_CONV_IN = np.array([[0], [1], [200], [0.03], [4]])
    REF_UPDATED_CONV_OUT = np.array([[0.0, 2.0, 0.04, 600, 8], [10, 12, 0.14, 1600, 18]])
    REF_UPDATED_BIAS_IN = np.array([2, 4, 600, 0.08, 10])

    @pytest.mark.parametrize("bias_in_value", [np.array([2, 4, 6, 8, 10])])
    def test_align_scales(self, bias_in_value):
        def check_updated_values(updated_conv_in, updated_conv_out, updated_bias_in):
            assert updated_conv_in.shape == self.REF_UPDATED_CONV_IN.shape
            assert np.allclose(updated_conv_in, self.REF_UPDATED_CONV_IN)
            assert updated_conv_out.shape == self.REF_UPDATED_CONV_OUT.shape
            assert np.allclose(updated_conv_out, self.REF_UPDATED_CONV_OUT)
            assert updated_bias_in.shape == self.REF_UPDATED_BIAS_IN.shape
            assert np.allclose(updated_bias_in, self.REF_UPDATED_BIAS_IN)

        conv_in_value = np.arange(5).reshape(5, 1)
        conv_out_value = np.arange(10).reshape(2, 5) * 2
        ascale = np.array([-5.0, 0.0, 1e-3, 1e3, 2])
        eps = 1e-10
        # Check nothing will happen if dims are wrong
        dims_descriptor = LayoutDescriptor(1, 0, 0)
        updated_conv_in, updated_conv_out, updated_bias_in = ChannelAlignment._align_scales(
            conv_in_value, conv_out_value, bias_in_value, ascale, dims_descriptor, dims_descriptor, eps
        )
        assert updated_conv_in is conv_in_value
        assert updated_conv_out is conv_out_value
        assert updated_bias_in is bias_in_value

        dims_descriptor = LayoutDescriptor(0, 1, 0)
        updated_conv_in, updated_conv_out, updated_bias_in = ChannelAlignment._align_scales(
            conv_in_value, conv_out_value, bias_in_value, ascale, dims_descriptor, dims_descriptor, eps
        )
        check_updated_values(updated_conv_in, updated_conv_out, updated_bias_in)

        # Check group conv producer case
        conv_in_value = conv_in_value.reshape(1, 5, 1)
        dims_descriptor_in = LayoutDescriptor(1, 2, 0)
        dims_descriptor_out = LayoutDescriptor(0, 1, 0)
        updated_conv_in, updated_conv_out, updated_bias_in = ChannelAlignment._align_scales(
            conv_in_value, conv_out_value, bias_in_value, ascale, dims_descriptor_in, dims_descriptor_out, eps
        )
        updated_conv_in = updated_conv_in.reshape(updated_conv_in.shape[1:])
        check_updated_values(updated_conv_in, updated_conv_out, updated_bias_in)

    GET_NODES_TEST_CASES = [(VALID_CONV_LAYER_ATTR, VALID_CONV_LAYER_ATTR, True)]
    GET_NODES_TEST_CASES.extend([(attr, VALID_CONV_LAYER_ATTR, True) for attr in INVALID_CONSUMER_CONV_LAYER_ATTRS])
    GET_NODES_TEST_CASES.extend([(VALID_CONV_LAYER_ATTR, attr, False) for attr in INVALID_CONSUMER_CONV_LAYER_ATTRS])
    GET_NODES_TEST_CASES.extend(
        [
            (INVALID_CONV_LAYER_ATTR, VALID_CONV_LAYER_ATTR, True),
            (VALID_CONV_LAYER_ATTR, INVALID_CONV_LAYER_ATTR, False),
            (INVALID_CONV_LAYER_ATTR, INVALID_CONV_LAYER_ATTR, False),
        ]
    )
    GET_NODES_TEST_CASES.extend([(attr, VALID_CONV_LAYER_ATTR, True) for attr in MATMUL_LAYER_METATYPES])
    GET_NODES_TEST_CASES.append((None, VALID_CONV_LAYER_ATTR, False))

    @pytest.mark.parametrize("first_conv_attrs,second_conv_attrs,ref_match", GET_NODES_TEST_CASES)
    def test_get_node_pairs(self, first_conv_attrs, second_conv_attrs, ref_match):
        algorithm = ChannelAlignment()
        algorithm._backend_entity = self.get_backend_cls()
        if first_conv_attrs is not None:
            first_conv_attrs = self.convert_conv_layer_attrs(first_conv_attrs)
        if second_conv_attrs is not None:
            second_conv_attrs = self.convert_conv_layer_attrs(second_conv_attrs)
        nncf_graph = NNCFGraphCA(
            self.get_conv_metatype(),
            conv_layer_attrs=first_conv_attrs,
            conv_2_layer_attrs=second_conv_attrs,
            use_one_layer_attrs=False,
        )
        pairs = algorithm._get_node_pairs(nncf_graph.nncf_graph)
        if ref_match:
            assert len(pairs) == 1
            conv_in, add_in, conv_out = pairs[0]
            assert conv_in.node_name == "/Conv_1_0"
            assert add_in is None
            assert conv_out.node_name == "/Conv_2_0"
        else:
            assert len(pairs) == 0

    def _get_nncf_graph(
        self, num_biases: int, conv_layer_attrs=DEPTHWISE_CONV_LAYER_ATTR, conv_layer_attrs_1=VALID_CONV_LAYER_ATTR
    ) -> NNCFGraph:
        cla = self.convert_conv_layer_attrs(conv_layer_attrs)
        cla_1 = self.convert_conv_layer_attrs(conv_layer_attrs_1)

        if num_biases == 0:
            return NNCFGraphCA(self.get_conv_metatype(), cla, cla_1).nncf_graph
        bla = self.get_add_layer_attrs()
        if num_biases == 1:
            return NNCFGraphCAWithBias(
                self.get_conv_metatype(),
                self.get_add_metatype(),
                cla,
                cla_1,
                both_biases=False,
                constant_metatype=self.get_constant_metatype(),
                add_layer_attrs=bla,
            ).nncf_graph
        return NNCFGraphCAWithBias(
            self.get_conv_metatype(),
            self.get_add_metatype(),
            cla,
            cla_1,
            both_biases=True,
            add_layer_attrs=bla,
            constant_metatype=self.get_constant_metatype(),
        ).nncf_graph

    @staticmethod
    def _get_constant_lambda(value, counter=False):
        if counter:
            _state = 0

        def f(*args, **kwargs):
            if not counter:
                return value
            nonlocal _state
            _state += 1
            return value + str(_state)

        return f

    @pytest.mark.parametrize("one_dim_mm", [False, True])
    @pytest.mark.parametrize("empty_statistics", [False, True])
    @pytest.mark.parametrize("num_biases", [0, 1, 2])
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-branches
    def test_transformation_layout(self, one_dim_mm, empty_statistics, num_biases, mocker):
        mocked_transformer = mocker.MagicMock()
        self.mock_model_transformer_factory(mocker, mocked_transformer)

        # NNCFGraph building
        first_conv_layer_attrs = DEPTHWISE_CONV_LAYER_ATTR if not one_dim_mm else MATMUL_LAYER_METATYPES[1]
        nncf_graph = self._get_nncf_graph(num_biases, first_conv_layer_attrs)
        self.mock_nncf_graph_factory(mocker, nncf_graph)

        self.mock_command_creation_factory(mocker)

        # Statistic points setup
        statistic_points = StatisticPointsContainer()
        target_node_name = "/Add_1_0" if num_biases else "/Conv_1_0"
        target_node = nncf_graph.get_node_by_name(target_node_name)
        backend_cls = self.get_backend_cls()
        ref_input_port_id, _ = backend_cls.get_activation_port_ids_for_node(target_node)
        target_point = self.target_point(TargetType.POST_LAYER_OPERATION, target_node_name, ref_input_port_id)

        class TestTensorStats(MinMaxTensorStatistic):
            @staticmethod
            def tensor_eq(*args, **kwargs):
                return True

        algorithm = ChannelAlignment()
        tensor_collector = TensorCollector()
        if empty_statistics:
            stat_value = None, None
        else:
            stat_value = (np.array([-1], dtype=np.int32), np.array([2], dtype=np.int32))

        tensor_collector.get_statistics = self._get_constant_lambda(TestTensorStats(*stat_value))
        statistic_points.add_statistic_point(StatisticPoint(target_point, tensor_collector, algorithm._algorithm_key))

        # Backend setup
        class MockBackend(backend_cls):
            pass

        ref_weights_val = "ref_weights_val"
        MockBackend.get_weight_value = self._get_constant_lambda(ref_weights_val, True)
        ref_bias_val = "ref_bias_val"
        MockBackend.get_bias_value = self._get_constant_lambda(ref_bias_val, True)

        # ConvParams setup
        ref_dims_in = LayoutDescriptor(0, 1, -1)
        ref_dims_out = LayoutDescriptor(0, 2, 1)
        if one_dim_mm:
            ref_dims_in = LayoutDescriptor(None, 1, -1)
        iter_ = (dims for dims in (ref_dims_in, ref_dims_out))

        def dims_iter(*args, **kwargs):
            return next(iter_)

        MockBackend.get_dims_descriptor = dims_iter

        # Algorithm fucntions mocking
        algorithm._backend_entity = MockBackend
        algorithm._set_backend_entity = mocker.MagicMock()
        ref_bias_in_after_align = "ref_bias_in_after_align"
        ref_bias_out_after_align = "ref_bias_out_after_align"
        algorithm._align_means = mocker.MagicMock(return_value=(ref_bias_in_after_align, ref_bias_out_after_align))
        ref_weights_in_after_scale_align = "ref_weights_in_after_scale_align"
        ref_weights_out_after_scale_align = "ref_weights_in_after_scale_align "
        ref_bias_in_after_scale_align = "ref_bias_in_after_scale_align" if num_biases > 1 else None
        algorithm._align_scales = mocker.MagicMock(
            return_value=(
                ref_weights_in_after_scale_align,
                ref_weights_out_after_scale_align,
                ref_bias_in_after_scale_align,
            )
        )
        algorithm.apply(None, nncf_graph, statistic_points)

        if empty_statistics or one_dim_mm:
            assert algorithm._align_means.call_count == 0
            assert algorithm._align_scales.call_count == 0
            mocked_transformer.transform.assert_called_once()
            arg = mocked_transformer.transform.call_args.args[0]
            assert len(arg.transformations) == 0
            return

        args = [
            np.zeros((1, 1, 1, 1)),
            np.zeros((1, 1, 1, 1)),
            ref_weights_val + "2",
            np.array(0.5, dtype=np.float32),
            ref_dims_out,
        ]
        for i in range(num_biases):
            args[i] = f"ref_bias_val{i + 1}"

        algorithm._align_means.assert_called_once_with(*args)

        assert algorithm._align_scales.call_count == 1
        args = algorithm._align_scales.call_args.args
        assert args[0] == ref_weights_val + "1"
        assert args[1] == ref_weights_val + "2"
        assert args[2] == ref_bias_in_after_align
        assert ((args[3] - 3) < EPS).all()
        assert args[4] == ref_dims_in
        assert args[5] == ref_dims_out
        assert args[6] < EPS

        mocked_transformer.transform.assert_called_once()
        arg = mocked_transformer.transform.call_args.args[0]
        transformations = arg.transformations

        target_names = {"/Conv_1_0": [], "/Conv_2_0": []}
        ref_values = {
            "/Conv_1_0": {
                "weight_value": ref_weights_in_after_scale_align,
                "bias_value": ref_bias_in_after_scale_align,
            },
            "/Conv_2_0": {"weight_value": ref_weights_out_after_scale_align, "bias_value": ref_bias_out_after_align},
        }

        bias_insert_cls, bias_update_cls, weights_update_cls = self.get_transformation_commands()
        for transformation in transformations:
            tp = transformation.target_point
            if isinstance(transformation, bias_update_cls):
                assert transformation.type == TransformationType.CHANGE
                _class = bias_update_cls
                _attr = "bias_value"
            elif isinstance(transformation, bias_insert_cls):
                assert transformation.type == TransformationType.INSERT
                _class = bias_insert_cls
                _attr = "bias_value"
            elif isinstance(transformation, weights_update_cls):
                assert transformation.type == TransformationType.CHANGE
                _class = weights_update_cls
                _attr = "weight_value"
            else:
                raise nncf.ValidationError(f"Wrong type of transformation: {type(transformation)}")

            target_names[tp.target_node_name].append(_class)
            assert ref_values[tp.target_node_name][_attr] == getattr(transformation, _attr)

        if num_biases == 2:
            ref_classes = {"/Conv_1_0": bias_update_cls, "/Conv_2_0": bias_update_cls}
        elif num_biases == 1:
            ref_classes = {"/Conv_1_0": bias_update_cls, "/Conv_2_0": bias_insert_cls}
        else:
            ref_classes = {"/Conv_1_0": bias_insert_cls, "/Conv_2_0": bias_insert_cls}

        for node_name, _transformations in target_names.items():
            assert len(_transformations) == 2
            assert weights_update_cls in _transformations
            assert ref_classes[node_name] in _transformations

    @pytest.mark.parametrize("num_biases", [0, 1, 2])
    def test_get_statistic_points(self, num_biases, mocker):
        nncf_graph = self._get_nncf_graph(num_biases)
        self.mock_nncf_graph_factory(mocker, nncf_graph)

        ref_subset_size = "ref_subset_size"
        ref_inplace = "ref_inplace"
        algorithm = ChannelAlignment(ref_subset_size, ref_inplace)
        algorithm._set_backend_entity = mocker.MagicMock()
        backend_cls = self.get_backend_cls()
        ref_stat_collector = "ref_stat_collector"

        class MockBackend(backend_cls):
            pass

        MockBackend.get_statistic_collector = mocker.MagicMock(return_value=ref_stat_collector)
        algorithm._backend_entity = MockBackend

        statistic_container = algorithm.get_statistic_points(None, nncf_graph)

        backend_cls = self.get_backend_cls()
        target_node_name = "/Add_1_0" if num_biases else "/Conv_1_0"
        target_node = nncf_graph.get_node_by_name(target_node_name)
        ref_input_port_id, _ = backend_cls.get_activation_port_ids_for_node(target_node)

        assert len(statistic_container) == 1
        assert target_node_name in statistic_container
        stat_points = statistic_container[target_node_name]
        assert len(stat_points) == 1

        assert len(stat_points[0].algorithm_to_tensor_collectors.keys()) == 1
        assert algorithm._algorithm_key in stat_points[0].algorithm_to_tensor_collectors
        tensor_collectors = stat_points[0].algorithm_to_tensor_collectors[algorithm._algorithm_key]
        assert len(tensor_collectors) == 1
        assert tensor_collectors[0] == ref_stat_collector
        MockBackend.get_statistic_collector.assert_called_once_with((2, 3), 1e-4, ref_subset_size, ref_inplace)

        target_point = stat_points[0].target_point
        assert target_point.target_node_name == target_node_name
        assert target_point.port_id == ref_input_port_id
        assert target_point.type == TargetType.POST_LAYER_OPERATION

    @pytest.mark.parametrize("inplace_ref", [False, True])
    @pytest.mark.parametrize("q_ref", [1e-4, 0.3])
    def test_statistic_collectors(self, inplace_ref, q_ref):
        reduction_axes_ref = (0, 2, 3)
        num_samples_ref = 123
        statistic_collector: TensorCollector = self.get_backend_cls().get_statistic_collector(
            reduction_axes=reduction_axes_ref, q=q_ref, num_samples=num_samples_ref, inplace=inplace_ref
        )

        assert len(statistic_collector.reducers) == 1
        reducer = statistic_collector.reducers.pop()
        assert isinstance(reducer, QuantileReducer)
        assert reducer._reduction_axes == reduction_axes_ref
        assert np.allclose(reducer._quantile, (q_ref, 1 - q_ref))

        assert len(statistic_collector.aggregators) == 2
        for aggr in statistic_collector.aggregators.values():
            assert isinstance(aggr, MedianAggregator)
            assert aggr.num_samples == num_samples_ref
            assert aggr._keepdims
            assert aggr._aggregation_axes == (0, 1)
