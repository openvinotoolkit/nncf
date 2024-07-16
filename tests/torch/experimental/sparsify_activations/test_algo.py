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

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import openvino as ov
import pytest
import torch
import torch.nn as nn

import nncf
import nncf.experimental
import nncf.experimental.torch.sparsify_activations
from nncf.experimental.torch.sparsify_activations.torch_backend import ACTIVATIONS_SPARSIFIER_PREFIX
from nncf.experimental.torch.sparsify_activations.torch_backend import ActivationsSparsifier
from nncf.scopes import IgnoredScope
from nncf.torch.nncf_network import NNCFNetwork
from tests.shared.nx_graph import compare_nx_graph_with_reference
from tests.shared.paths import TEST_ROOT
from tests.torch.experimental.sparsify_activations.helpers import TwoLinearModel
from tests.torch.experimental.sparsify_activations.helpers import dummy_llama_model
from tests.torch.helpers import set_torch_seed


@dataclass
class AlgoTestDesc:
    model_name: str
    model_getter: Callable[[], nn.Module]
    dataset_getter: Callable[[torch.device], nncf.Dataset]
    compress_weights: bool
    target_sparsity_by_scope: Dict[str, float]
    ignored_scope: nncf.IgnoredScope | None
    ref_sparsifier_target_sparsity: Dict[str, float]
    ref_num_batches_tracked: int

    @property
    def desc_id(self):
        compress_weights_flag = "_compressed_weights" if self.compress_weights else ""
        return f"{self.model_name}{compress_weights_flag}"

    @property
    def ref_dot_path(self):
        return TEST_ROOT / Path("torch/data/sparsify_activations", f"{self.desc_id}.dot")


AlgoTestDescs: list[AlgoTestDesc] = []
for compress_weights in [False, True]:
    AlgoTestDescs += [
        AlgoTestDesc(
            model_name="linear",
            model_getter=lambda: nn.Linear(4, 2),
            dataset_getter=lambda device: nncf.Dataset(torch.randn([3, 2, 4]).to(device)),
            compress_weights=compress_weights,
            target_sparsity_by_scope={
                "{re}.*linear.*": 0.3,
            },
            ignored_scope=None,
            ref_sparsifier_target_sparsity={
                f"{ACTIVATIONS_SPARSIFIER_PREFIX}_Linear/linear_0": 0.3,
            },
            ref_num_batches_tracked=3,
        ),
        AlgoTestDesc(
            model_name="two_linear",
            model_getter=TwoLinearModel,
            dataset_getter=lambda device: nncf.Dataset(torch.randint(0, 30, (3, 2, 8)).to(device)),
            compress_weights=compress_weights,
            target_sparsity_by_scope={
                "{re}.*linear2.*": 0.4,
            },
            ignored_scope=IgnoredScope(patterns=[".*linear1.*"]),
            ref_sparsifier_target_sparsity={
                f"{ACTIVATIONS_SPARSIFIER_PREFIX}_TwoLinearModel/Linear[linear2]/linear_0": 0.4,
            },
            ref_num_batches_tracked=3,
        ),
        AlgoTestDesc(
            model_name="dummy_llama",
            model_getter=dummy_llama_model,
            dataset_getter=lambda device: nncf.Dataset(torch.randint(0, 30, (3, 2, 8)).to(device)),
            compress_weights=compress_weights,
            target_sparsity_by_scope={
                "{re}.*gate_proj.*": 0.2,
                "{re}.*up_proj.*": 0.3,
                "{re}.*down_proj.*": 0.4,
            },
            ignored_scope=None,
            ref_sparsifier_target_sparsity={
                (
                    f"{ACTIVATIONS_SPARSIFIER_PREFIX}_LlamaForCausalLM/LlamaModel[model]/ModuleList[layers]/"
                    f"LlamaDecoderLayer[{layer_id}]/LlamaMLP[mlp]/Linear[{name}]/linear_0"
                ): sparsity
                for name, sparsity in [("gate_proj", 0.2), ("up_proj", 0.3), ("down_proj", 0.4)]
                for layer_id in [0, 1]
            },
            ref_num_batches_tracked=3,
        ),
    ]


@pytest.mark.parametrize("desc", AlgoTestDescs, ids=[p.desc_id for p in AlgoTestDescs], scope="class")
@pytest.mark.parametrize("use_cuda", [False, True], ids=["cpu", "cuda"], scope="class")
class TestSparsifyActivationsAlgorithm:

    @pytest.fixture(autouse=True, scope="class")
    def setup(self, request, desc: AlgoTestDesc, use_cuda: bool):
        if use_cuda and not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        request.cls.use_cuda = use_cuda
        device = torch.device("cuda" if use_cuda else "cpu")
        request.cls.device = device
        request.cls.desc = desc
        with set_torch_seed():
            model = desc.model_getter()
            model = model.to(device).eval()
            dataset = desc.dataset_getter(device)
            if desc.compress_weights:
                model = nncf.compress_weights(
                    model,
                    mode=nncf.CompressWeightsMode.INT8_SYM,
                    dataset=dataset,
                )
            model = nncf.experimental.torch.sparsify_activations.sparsify_activations(
                model=model,
                dataset=dataset,
                target_sparsity_by_scope=desc.target_sparsity_by_scope,
                ignored_scope=desc.ignored_scope,
            )
        request.cls.model = model
        request.cls.dataset = dataset

    def test_inserted_sparsifier(self):
        desc: AlgoTestDesc = self.desc
        model = self.model
        assert isinstance(model, NNCFNetwork)
        num_sparsifiers = 0
        for name, op in model.nncf.external_op.items():
            if isinstance(op, ActivationsSparsifier):
                assert op.target_sparsity == desc.ref_sparsifier_target_sparsity[name]
                assert op.num_batches_tracked == desc.ref_num_batches_tracked
                num_sparsifiers += 1
        assert num_sparsifiers == len(desc.ref_sparsifier_target_sparsity)

    def test_nncf_graph(self):
        desc: AlgoTestDesc = self.desc
        model: NNCFNetwork = self.model
        graph = model.nncf.get_graph()
        graph.dump_graph(desc.ref_dot_path)
        graph = model.nncf.get_graph().get_graph_for_structure_analysis()
        compare_nx_graph_with_reference(graph, desc.ref_dot_path)

    def test_export_openvino(self):
        model: NNCFNetwork = self.model
        example_input = next(iter(self.dataset.get_inference_data()))
        with torch.no_grad():
            torch_outputs = model(example_input)
            if isinstance(torch_outputs, dict):
                torch_outputs = tuple(torch_outputs.values())
            if not isinstance(torch_outputs, tuple):
                torch_outputs = (torch_outputs,)

        ov_model = ov.convert_model(model, example_input=example_input)
        compiled_model = ov.compile_model(ov_model, "CPU")
        ov_outputs = compiled_model(example_input.cpu()).to_tuple()

        assert len(torch_outputs) == len(ov_outputs)
        for torch_output, ov_output in zip(torch_outputs, ov_outputs):
            torch.testing.assert_close(
                torch_output.cpu(),
                torch.from_numpy(ov_output),
                rtol=1e-3,
                atol=1e-3,
            )
