
import pytest

from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizerGroup
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantizationParameters
from nncf.quantization.algorithms.definitions import Granularity
from tests.common.quantization.mock_graphs import NodeWithType
from tests.common.quantization.metatypes import Conv2dTestMetatype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.common.graph.operator_metatypes import InputNoopMetatype
from tests.common.quantization.test_filter_constant_nodes import create_mock_graph
from tests.common.quantization.test_filter_constant_nodes import get_nncf_graph_from_mock_nx_graph
from nncf.quantization.algorithms.min_max.onnx_backend import \
    ONNXMinMaxAlgoBackend


class NNCFGraphToTest:
    def __init__(self):
        #       Original graph
        #          Input_1
        #             |
        #           Conv_1
        #             |
        #           Output_1
        nodes = [NodeWithType('Input_1', InputNoopMetatype),
                 NodeWithType('Conv_1', Conv2dTestMetatype),
                 NodeWithType('Output_1', OutputNoopMetatype),
                 ]
        node_edges = [('Input_1', 'Conv_1'), ('Conv_1', 'Output_1')]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)


@pytest.mark.parametrize('weight_granularity', [Granularity.PERCHANNEL, Granularity.PERTENSOR])
@pytest.mark.parametrize('activation_granularity', [Granularity.PERCHANNEL, Granularity.PERTENSOR])
@pytest.mark.parametrize('preset', [QuantizationPreset.MIXED, QuantizationPreset.PERFORMANCE])
@pytest.mark.parametrize('weight_bits', [8, 4, 1])
@pytest.mark.parametrize('activation_bits', [8, 4, 1])
@pytest.mark.parametrize('weight_signedness_to_force', [True, False, None])
@pytest.mark.parametrize('activation_signedness_to_force', [True, False, None])
@pytest.mark.parametrize('nncf_graph', [NNCFGraphToTest()])
def test_quantizer_config_from_min_max_params(weight_granularity, activation_granularity, preset, weight_bits,
                                              activation_bits,
                                              weight_signedness_to_force, activation_signedness_to_force, nncf_graph):
    min_max_algo = MinMaxQuantization(
        MinMaxQuantizationParameters(weight_granularity=weight_granularity,
                                     activation_granularity=activation_granularity))
    min_max_algo._backend_entity = ONNXMinMaxAlgoBackend()
    q_setup = min_max_algo._get_quantizer_setup(nncf_graph.nncf_graph)
    q_g_to_quantization_mode = {}
    for q_g in QuantizerGroup:
        q_g_to_quantization_mode[q_g] = preset.get_params_configured_by_preset(q_g)['mode']
    for quantization_point in q_setup.quantization_points.values():
        if quantization_point.is_weight_quantization_point():
            assert quantization_point.qconfig.mode == q_g_to_quantization_mode[QuantizerGroup.WEIGHTS]
            assert quantization_point.qconfig.per_channel == (weight_granularity == Granularity.PERCHANNEL)
            assert quantization_point.qconfig.num_bits == weight_bits
            assert quantization_point.qconfig.signedness_to_force == weight_signedness_to_force
        if quantization_point.is_activation_quantization_point():
            assert quantization_point.qconfig.per_channel == (activation_granularity == Granularity.PERCHANNEL)
            assert quantization_point.qconfig.num_bits == activation_bits
            assert quantization_point.qconfig.mode == q_g_to_quantization_mode[QuantizerGroup.ACTIVATIONS]
            assert quantization_point.qconfig.signedness_to_force == activation_signedness_to_force
