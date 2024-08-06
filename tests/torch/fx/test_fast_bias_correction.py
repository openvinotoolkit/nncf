# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import pytest
import torch
import torch.fx
from torch._export import capture_pre_autograd_graph

from nncf.common.factory import NNCFGraphFactory
from nncf.experimental.torch.fx.transformations import apply_quantization_transformations
from nncf.quantization.algorithms.fast_bias_correction.torch_fx_backend import FXFastBiasCorrectionAlgoBackend
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching
from nncf.torch.model_graph_manager import OPERATORS_WITH_BIAS_METATYPES
from tests.post_training.test_templates.test_fast_bias_correction import TemplateTestFBCAlgorithm


class TestTorchFXFBCAlgorithm(TemplateTestFBCAlgorithm):
    @staticmethod
    def list_to_backend_type(data: List) -> torch.Tensor:
        return torch.Tensor(data)

    @staticmethod
    def get_backend() -> FXFastBiasCorrectionAlgoBackend:
        return FXFastBiasCorrectionAlgoBackend

    def _get_fx_model(model: torch.nn.Module) -> torch.fx.GraphModule:
        device = next(model.named_parameters())[1].device
        input_shape = model.INPUT_SIZE
        if input_shape is None:
            input_shape = [1, 3, 32, 32]
        ex_input = torch.ones(input_shape).to(device)
        model.eval()
        with disable_patching():
            fx_model = capture_pre_autograd_graph(model, args=(ex_input,))
        apply_quantization_transformations(fx_model)
        return fx_model

    @staticmethod
    def backend_specific_model(model: torch.nn.Module, tmp_dir: str) -> torch.fx.GraphModule:
        fx_model = TestTorchFXFBCAlgorithm._get_fx_model(model)
        return fx_model

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
    def check_bias(model: torch.fx.GraphModule, ref_bias: list):
        ref_bias = torch.Tensor(ref_bias)
        nncf_graph = NNCFGraphFactory.create(model)
        for node in nncf_graph.get_all_nodes():
            if node.metatype not in OPERATORS_WITH_BIAS_METATYPES:
                continue
            bias_value = FXFastBiasCorrectionAlgoBackend.get_bias_value(node, nncf_graph, model)
            bias_value = torch.flatten(bias_value.data).cpu()
            # TODO(AlexanderDokuchaev): return atol=0.0001 after fix 109189
            assert torch.all(torch.isclose(bias_value, ref_bias, atol=0.02)), f"{bias_value} != {ref_bias}"
            return
        raise ValueError("Not found node with bias")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skipping for CPU-only setups")
class TestTorchFXCudaFBCAlgorithm(TestTorchFXFBCAlgorithm):
    @staticmethod
    def list_to_backend_type(data: List) -> torch.Tensor:
        return torch.Tensor(data).cuda()

    @staticmethod
    def backend_specific_model(model: bool, tmp_dir: str):
        fx_cuda_model = TestTorchFXFBCAlgorithm._get_fx_model(model.cuda())
        return fx_cuda_model

    @staticmethod
    def fn_to_type(tensor):
        return torch.Tensor(tensor).cuda()

    @staticmethod
    def check_bias(model: torch.fx.GraphModule, ref_bias: list):
        TestTorchFXFBCAlgorithm.check_bias(model, ref_bias)
