import pytest
from nncf.common.hardware.config import HWConfigType
from nncf.experimental.onnx.statistics.collectors import ONNXMeanMinMaxStatisticCollector
from nncf.experimental.onnx.statistics.collectors import ONNXMinMaxStatisticCollector
from nncf.quantization.algorithms.definitions import RangeType
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantizationParameters
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.min_max.onnx_backend import \
    ONNXMinMaxAlgoBackend
from tests.onnx.models import LinearModel
from tests.onnx.quantization.test_quantizer_config import NNCFGraphToTest


@pytest.mark.parametrize('target_device', [HWConfigType.CPU, HWConfigType.GPU, HWConfigType.VPU])
def test_target_device(target_device):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters(target_device=target_device))
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = ONNXMinMaxAlgoBackend()
    assert min_max_algo._parameters.target_device == target_device


@pytest.mark.parametrize('range_type', [RangeType.MINMAX, RangeType.MEAN_MINMAX])
def test_range_type(range_type):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters(range_type=range_type))
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = ONNXMinMaxAlgoBackend()
    model = LinearModel().onnx_model
    assert min_max_algo._parameters.range_type == range_type
    stat_points = min_max_algo.get_statistic_points(model)

    for node_name, stat_point in stat_points.items():
        for stat_point_ in stat_point:
            for tensor_collector in stat_point_.algorithm_to_tensor_collectors[MinMaxQuantization]:
                if range_type == RangeType.MINMAX:
                    assert isinstance(tensor_collector, ONNXMinMaxStatisticCollector)
                elif range_type == RangeType.MEAN_MINMAX:
                    assert isinstance(tensor_collector, ONNXMeanMinMaxStatisticCollector)


@pytest.mark.parametrize('quantize_outputs', [False, True])
def test_quantize_outputs(quantize_outputs):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters(quantize_outputs=quantize_outputs))
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = ONNXMinMaxAlgoBackend()
    nncf_graph = NNCFGraphToTest().nncf_graph
    assert min_max_algo._parameters.quantize_outputs == quantize_outputs
    q_setup = min_max_algo._get_quantizer_setup(nncf_graph)
    act_num_q, weight_num_q = 0, 0
    for quantization_point in q_setup.quantization_points.values():
        if quantization_point.is_activation_quantization_point():
            act_num_q += 1
        if quantization_point.is_weight_quantization_point():
            weight_num_q += 1

    if quantize_outputs:
        assert act_num_q == 2
    else:
        assert act_num_q == 1
    assert weight_num_q == 1


@pytest.mark.parametrize('ignored_scopes', [[], ['/Conv_1_0']])
def test_ignored_scopes(ignored_scopes):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters(ignored_scopes=ignored_scopes))
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = ONNXMinMaxAlgoBackend()
    nncf_graph = NNCFGraphToTest().nncf_graph
    assert min_max_algo._parameters.ignored_scopes == ignored_scopes
    q_setup = min_max_algo._get_quantizer_setup(nncf_graph)
    act_num_q, weight_num_q = 0, 0
    for quantization_point in q_setup.quantization_points.values():
        if quantization_point.is_activation_quantization_point():
            act_num_q += 1
        if quantization_point.is_weight_quantization_point():
            weight_num_q += 1

    if ignored_scopes:
        assert act_num_q == 0
    else:
        assert act_num_q == 1
    assert weight_num_q == 1
