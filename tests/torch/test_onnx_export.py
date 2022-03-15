import pytest
from typing import Any
import torch
import onnx

from nncf import NNCFConfig
from nncf.torch.exporter import PTExporter
from tests.torch.helpers import MockModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
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


# pylint: disable=protected-access
@pytest.mark.parametrize('save_format, refs',
                         (
                             ('onnx', ('onnx', {'opset_version': PTExporter._ONNX_DEFAULT_OPSET})),
                             ('onnx_9', ('onnx', {'opset_version': 9})),
                             ('onnx_10', ('onnx', {'opset_version': 10})),
                             ('onnx_11', ('onnx', {'opset_version': 11})),
                             ('onnx_0', ValueError),
                             ('onnx_onnx', ValueError),
                         )
                        )
def test_exporter_parser_format(save_format: str, refs: Any):
    # pylint: disable=broad-except
    try:
        save_format, args = PTExporter.parse_format(save_format)
    except Exception as e:
        if not isinstance(refs, tuple):
            assert isinstance(e, refs)
            return

    assert save_format == refs[0]
    assert args == refs[1]

@pytest.mark.parametrize('save_format, ref_opset',
                         (
                             ('onnx', 10),
                             ('onnx_9', 9),
                             ('onnx_11', 11)
                         )
                        )
def test_exported_version(tmp_path: str, save_format: str, ref_opset: int):
    model = MockModel()
    config = NNCFConfig()
    config.update({"input_info": {"sample_size": [1, 1, 1, 1]}})

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    onnx_checkpoint_path = tmp_path / 'model.onnx'
    compression_ctrl.export_model(onnx_checkpoint_path, save_format)
    model_proto = onnx.load_model(onnx_checkpoint_path)
    # pylint: disable=no-member
    assert model_proto.opset_import[0].version == ref_opset
