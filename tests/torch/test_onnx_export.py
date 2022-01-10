import torch

from nncf import NNCFConfig
from tests.torch.helpers import get_nodes_by_type
from tests.torch.helpers import load_exported_onnx_version


class ModelForIONamingTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 1, 1)
        self.linear = torch.nn.Linear(1, 1)
        self.embedding = torch.nn.Embedding(1, 1)

    def forward(self, conv_input, linear_input, embedding_input):
        return [self.conv(conv_input),
                {
                    'linear': self.linear(linear_input),
                    'embedding': self.embedding(embedding_input)
                }]


def test_io_nodes_naming_scheme(tmp_path):
    config = NNCFConfig.from_dict({
        "input_info": [
            {
                "sample_size": [1, 1, 1],
            },
            {
                "sample_size": [1, 1],
            },
            {
                "sample_size": [1, 1],
                "type": "long",
                "filler": "zeros"
            },
        ]
    })
    onnx_model_proto = load_exported_onnx_version(config, ModelForIONamingTest(), tmp_path)
    conv_node = next(iter(get_nodes_by_type(onnx_model_proto, "Conv")))
    linear_node = next(iter(get_nodes_by_type(onnx_model_proto, "Gemm")))
    embedding_node = next(iter(get_nodes_by_type(onnx_model_proto, "Gather")))

    for idx, node in enumerate([conv_node, linear_node, embedding_node]):
        input_tensor_ids = [x for x in node.input if "input" in x]
        assert len(input_tensor_ids) == 1
        assert input_tensor_ids[0] == f"input.{idx}"

        assert len(node.output) == 1
        assert node.output[0] == f"output.{idx}"
