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

from typing import List

import pytest
import torch
import torch.fx

from nncf.common.factory import NNCFGraphFactory
from nncf.quantization.algorithms.fast_bias_correction.torch_fx_backend import FXFastBiasCorrectionAlgoBackend
from nncf.torch.model_graph_manager import OPERATORS_WITH_BIAS_METATYPES
from tests.cross_fw.test_templates.test_fast_bias_correction import TemplateTestFBCAlgorithm
from tests.torch.fx.helpers import get_torch_fx_model_q_transformed


class TestTorchFXFBCAlgorithm(TemplateTestFBCAlgorithm):
    @staticmethod
    def list_to_backend_type(data: List) -> torch.Tensor:
        return torch.Tensor(data)

    @staticmethod
    def get_backend() -> FXFastBiasCorrectionAlgoBackend:
        return FXFastBiasCorrectionAlgoBackend

    @staticmethod
    def backend_specific_model(model: torch.nn.Module, tmp_dir: str) -> torch.fx.GraphModule:
        fx_model = get_torch_fx_model_q_transformed(model, torch.ones(model.INPUT_SIZE))
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
        return get_torch_fx_model_q_transformed(model.cuda(), torch.ones(model.INPUT_SIZE))

    @staticmethod
    def fn_to_type(tensor):
        return torch.Tensor(tensor).cuda()

    @staticmethod
    def check_bias(model: torch.fx.GraphModule, ref_bias: list):
        TestTorchFXFBCAlgorithm.check_bias(model, ref_bias)
