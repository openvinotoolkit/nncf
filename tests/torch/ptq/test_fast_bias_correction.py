# Copyright (c) 2023 Intel Corporation
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

import torch

from nncf.common.factory import NNCFGraphFactory
from nncf.quantization.algorithms.fast_bias_correction.torch_backend import PTFastBiasCorrectionAlgoBackend
from nncf.torch.model_analyzer import get_fused_bias_value
from nncf.torch.model_analyzer import is_node_with_fused_bias
from nncf.torch.nncf_network import NNCFNetwork
from tests.post_training.test_templates.test_fast_bias_correction import TemplateTestFBCAlgorithm
from tests.torch.ptq.helpers import get_nncf_network


class TestTorchFBCAlgorithm(TemplateTestFBCAlgorithm):
    @staticmethod
    def list_to_backend_type(data: List) -> torch.Tensor:
        return torch.Tensor(data)

    @staticmethod
    def get_backend() -> PTFastBiasCorrectionAlgoBackend:
        return PTFastBiasCorrectionAlgoBackend

    @staticmethod
    def backend_specific_model(model: bool, tmp_dir: str):
        return get_nncf_network(model, model.INPUT_SIZE)

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
    def check_bias(model: NNCFNetwork, ref_bias: list):
        ref_bias = torch.Tensor(ref_bias)
        nncf_graph = NNCFGraphFactory.create(model)
        for node in nncf_graph.get_all_nodes():
            if not is_node_with_fused_bias(node, model):
                continue
            bias_value = get_fused_bias_value(node, model)
            # TODO(AlexanderDokuchaev): return atol=0.0001 after fix 109189
            assert torch.all(torch.isclose(bias_value, ref_bias, atol=0.02)), f"{bias_value} != {ref_bias}"
            return
        raise ValueError("Not found node with bias")
