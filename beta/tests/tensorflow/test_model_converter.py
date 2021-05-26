from nncf.common.graph.graph import NNCFGraphNodeType
from beta.nncf.tensorflow.graph.converter import convert_keras_model_to_nncf_graph
from beta.tests.tensorflow.helpers import get_basic_conv_test_model
from beta.tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from beta.tests.tensorflow.quantization.test_algorithm_quantization import get_basic_quantization_config


def test_struct_auxiliary_nodes_nncf_graph():
    model = get_basic_conv_test_model()
    config = get_basic_quantization_config()
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)

    nncf_graph = convert_keras_model_to_nncf_graph(compressed_model)

    input_nodes = nncf_graph.get_input_nodes()
    output_nodes = nncf_graph.get_output_nodes()

    assert len(input_nodes) == 1
    assert len(output_nodes) == 1

    assert input_nodes[0].node_type == NNCFGraphNodeType.INPUT_NODE
    assert output_nodes[0].node_type == NNCFGraphNodeType.OUTPUT_NODE
