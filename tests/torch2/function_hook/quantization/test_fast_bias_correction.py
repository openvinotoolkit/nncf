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


import pytest
import torch

from nncf.quantization.algorithms.fast_bias_correction.torch_backend import PTFastBiasCorrectionAlgoBackend
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import GraphModelWrapper
from nncf.torch.function_hook.wrapper import wrap_model
from nncf.torch.model_graph_manager import get_fused_bias_value
from nncf.torch.model_graph_manager import is_node_with_fused_bias
from tests.cross_fw.test_templates.test_fast_bias_correction import TemplateTestFBCAlgorithm


class TestTorchFBCAlgorithm(TemplateTestFBCAlgorithm):
    @staticmethod
    def list_to_backend_type(data: list) -> torch.Tensor:
        return torch.Tensor(data)

    @staticmethod
    def get_backend() -> PTFastBiasCorrectionAlgoBackend:
        return PTFastBiasCorrectionAlgoBackend

    @staticmethod
    def backend_specific_model(model: bool, tmp_dir: str):
        return GraphModelWrapper(wrap_model(model), torch.ones(model.INPUT_SIZE))

    @staticmethod
    def fn_to_type(tensor):
        return torch.Tensor(tensor)

    @staticmethod
    def get_transform_fn():
        def transform_fn(data_item):
            tensor, _ = data_item
            return tensor

        return transform_fn

    @staticmethod
    def check_bias(model: GraphModelWrapper, ref_bias: list):
        ref_bias = torch.Tensor(ref_bias)
        nncf_graph = model.get_graph()
        for node in nncf_graph.get_all_nodes():
            if not is_node_with_fused_bias(node, nncf_graph):
                continue
            bias_value = get_fused_bias_value(node, nncf_graph, model.model).cpu()
            # TODO(AlexanderDokuchaev): return atol=0.0001 after fix 109189
            assert torch.all(torch.isclose(bias_value, ref_bias, atol=0.02)), f"{bias_value} != {ref_bias}"
            return
        msg = "Not found node with bias"
        raise ValueError(msg)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skipping for CPU-only setups")
class TestTorchCudaFBCAlgorithm(TestTorchFBCAlgorithm):
    @staticmethod
    def list_to_backend_type(data: list) -> torch.Tensor:
        return torch.Tensor(data).cuda()

    @staticmethod
    def backend_specific_model(model: bool, tmp_dir: str):
        return GraphModelWrapper(wrap_model(model.cuda()), torch.ones(model.INPUT_SIZE).cuda())

    @staticmethod
    def fn_to_type(tensor):
        return torch.Tensor(tensor).cuda()
