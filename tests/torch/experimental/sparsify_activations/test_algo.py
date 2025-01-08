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

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import openvino as ov
import pytest
import torch
import torch.nn as nn

import nncf
import nncf.experimental
import nncf.experimental.torch.sparsify_activations
from nncf.experimental.torch.sparsify_activations.sparsify_activations_impl import SparsifyActivationsAlgorithm
from nncf.experimental.torch.sparsify_activations.sparsify_activations_impl import TargetScope
from nncf.experimental.torch.sparsify_activations.torch_backend import ACTIVATIONS_SPARSIFIER_PREFIX
from nncf.experimental.torch.sparsify_activations.torch_backend import ActivationsSparsifier
from nncf.scopes import IgnoredScope
from nncf.torch.model_creation import wrap_model
from nncf.torch.nncf_network import NNCFNetwork
from tests.cross_fw.shared.nx_graph import compare_nx_graph_with_reference
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch.experimental.sparsify_activations.helpers import ThreeLinearModel
from tests.torch.experimental.sparsify_activations.helpers import count_sparsifier_patterns_in_ov
from tests.torch.experimental.sparsify_activations.helpers import dummy_llama_model
from tests.torch.helpers import set_torch_seed


@dataclass
class SparsifyActivationsAlgorithmTestDesc:
    name: str
    model_getter: Callable[[], nn.Module]
    dataset_getter: Callable[[torch.device], nncf.Dataset]
    target_sparsity_by_scope: Dict[TargetScope, float]
    ignored_scope: Optional[nncf.IgnoredScope]
    ref_sparsifier_target_sparsity: Dict[str, float]
    ref_num_batches_tracked: int
    ref_num_patterns_in_ov: int


sparsify_activations_algorithm_test_descs = [
    SparsifyActivationsAlgorithmTestDesc(
        name="linear",
        model_getter=lambda: nn.Linear(4, 2),
        dataset_getter=lambda device: nncf.Dataset(torch.randn([3, 2, 4]).to(device)),
        target_sparsity_by_scope={
            TargetScope(names=["Linear/linear_0"]): 0.3,
        },
        ignored_scope=None,
        ref_sparsifier_target_sparsity={
            f"{ACTIVATIONS_SPARSIFIER_PREFIX}_Linear/linear_0": 0.3,
        },
        ref_num_batches_tracked=3,
        ref_num_patterns_in_ov=1,
    ),
    SparsifyActivationsAlgorithmTestDesc(
        name="three_linear",
        model_getter=ThreeLinearModel,
        dataset_getter=lambda device: nncf.Dataset(torch.randint(0, 30, (3, 2, 8)).to(device)),
        target_sparsity_by_scope={
            TargetScope(types=["linear"]): 0.4,
        },
        ignored_scope=None,
        ref_sparsifier_target_sparsity={
            f"{ACTIVATIONS_SPARSIFIER_PREFIX}_ThreeLinearModel/Linear[linear1]/linear_0": 0.4,
            f"{ACTIVATIONS_SPARSIFIER_PREFIX}_ThreeLinearModel/Linear[linear2]/linear_0": 0.4,
            f"{ACTIVATIONS_SPARSIFIER_PREFIX}_ThreeLinearModel/Linear[linear3]/linear_0": 0.4,
        },
        ref_num_batches_tracked=3,
        ref_num_patterns_in_ov=2,  # Sparsifiers are combined in linear1 and linear2
    ),
    SparsifyActivationsAlgorithmTestDesc(
        name="three_linear_ignore1",
        model_getter=ThreeLinearModel,
        dataset_getter=lambda device: nncf.Dataset(torch.randint(0, 30, (3, 2, 8)).to(device)),
        target_sparsity_by_scope={
            TargetScope(names=["ThreeLinearModel/Linear[linear2]/linear_0"]): 0.4,
            TargetScope(patterns=[".*linear3.*"]): 0.4,
        },
        ignored_scope=IgnoredScope(patterns=[".*linear1.*"]),
        ref_sparsifier_target_sparsity={
            f"{ACTIVATIONS_SPARSIFIER_PREFIX}_ThreeLinearModel/Linear[linear2]/linear_0": 0.4,
            f"{ACTIVATIONS_SPARSIFIER_PREFIX}_ThreeLinearModel/Linear[linear3]/linear_0": 0.4,
        },
        ref_num_batches_tracked=3,
        ref_num_patterns_in_ov=2,
    ),
    SparsifyActivationsAlgorithmTestDesc(
        name="dummy_llama",
        model_getter=dummy_llama_model,
        dataset_getter=lambda device: nncf.Dataset(torch.randint(0, 30, (3, 2, 8)).to(device)),
        target_sparsity_by_scope={
            TargetScope(patterns=[".*gate_proj.*"]): 0.2,
            TargetScope(patterns=[".*up_proj.*"]): 0.3,
            TargetScope(patterns=[".*down_proj.*"]): 0.4,
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
        ref_num_patterns_in_ov=6,
    ),
]


@pytest.mark.parametrize(
    "desc",
    sparsify_activations_algorithm_test_descs,
    ids=[p.name for p in sparsify_activations_algorithm_test_descs],
    scope="class",
)
@pytest.mark.parametrize("compress_weights", [False, True], scope="class")
@pytest.mark.parametrize("use_cuda", [False, True], ids=["cpu", "cuda"], scope="class")
class TestSparsifyActivationsAlgorithm:

    @pytest.fixture(autouse=True, scope="class")
    def setup(self, request, desc: SparsifyActivationsAlgorithmTestDesc, compress_weights: bool, use_cuda: bool):
        if use_cuda and not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        request.cls.use_cuda = use_cuda
        device = torch.device("cuda" if use_cuda else "cpu")
        request.cls.device = device
        request.cls.desc = desc
        request.cls.compress_weights = compress_weights
        with set_torch_seed():
            model = desc.model_getter()
            model = model.to(device).eval()
            dataset = desc.dataset_getter(device)
            if compress_weights:
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
        desc: SparsifyActivationsAlgorithmTestDesc = self.desc
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
        desc: SparsifyActivationsAlgorithmTestDesc = self.desc
        model: NNCFNetwork = self.model
        file_name = "_".join(
            filter(None, [desc.name, "int8_sym_weights" if self.compress_weights else None, "sparse_activations"])
        )
        ref_dot_path = Path(TEST_ROOT, "torch", "data", "experimental", "sparsify_activations", f"{file_name}.dot")
        graph = model.nncf.get_graph().get_graph_for_structure_analysis()
        compare_nx_graph_with_reference(graph, ref_dot_path)

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
        assert count_sparsifier_patterns_in_ov(ov_model) == self.desc.ref_num_patterns_in_ov

        compiled_model = ov.compile_model(ov_model, "CPU", config={ov.properties.hint.inference_precision: "f32"})
        ov_outputs = compiled_model(example_input.cpu()).to_tuple()
        assert len(torch_outputs) == len(ov_outputs)
        for torch_output, ov_output in zip(torch_outputs, ov_outputs):
            torch.testing.assert_close(torch_output.cpu(), torch.from_numpy(ov_output), rtol=1e-3, atol=1e-3)


@dataclass
class TargetSparsityByNodeTestDesc:
    target_sparsity_by_scope: Dict[TargetScope, float]
    ignored_scope: IgnoredScope
    ref_target_sparsity_by_node_name: Optional[Dict[str, float]] = None
    raised_error_message: Optional[str] = None


@pytest.mark.parametrize(
    "desc",
    [
        TargetSparsityByNodeTestDesc(
            target_sparsity_by_scope={TargetScope(patterns=[".*linear.*"]): 0.3},
            ignored_scope=IgnoredScope(),
            ref_target_sparsity_by_node_name={
                "ThreeLinearModel/Linear[linear1]/linear_0": 0.3,
                "ThreeLinearModel/Linear[linear2]/linear_0": 0.3,
                "ThreeLinearModel/Linear[linear3]/linear_0": 0.3,
            },
        ),
        TargetSparsityByNodeTestDesc(
            target_sparsity_by_scope={TargetScope(patterns=[".*linear[23].*"], types=["linear"]): 0.3},
            ignored_scope=IgnoredScope(),
            ref_target_sparsity_by_node_name={
                "ThreeLinearModel/Linear[linear1]/linear_0": 0.3,
                "ThreeLinearModel/Linear[linear2]/linear_0": 0.3,
                "ThreeLinearModel/Linear[linear3]/linear_0": 0.3,
            },
        ),
        TargetSparsityByNodeTestDesc(
            target_sparsity_by_scope={
                TargetScope(
                    subgraphs=[nncf.Subgraph(inputs=["/nncf_model_input_0"], outputs=["/nncf_model_output_0"])]
                ): 0.1,
            },
            ignored_scope=IgnoredScope(),
            ref_target_sparsity_by_node_name={
                "ThreeLinearModel/Linear[linear1]/linear_0": 0.1,
                "ThreeLinearModel/Linear[linear3]/linear_0": 0.1,
            },
        ),
        TargetSparsityByNodeTestDesc(
            target_sparsity_by_scope={
                TargetScope(names=["ThreeLinearModel/Linear[linear1]/linear_0"]): 0.1,
                TargetScope(patterns=[".*linear[23].*"]): 0.3,
            },
            ignored_scope=IgnoredScope(patterns=[".*linear2.*"]),
            ref_target_sparsity_by_node_name={
                "ThreeLinearModel/Linear[linear1]/linear_0": 0.1,
                "ThreeLinearModel/Linear[linear3]/linear_0": 0.3,
            },
        ),
        TargetSparsityByNodeTestDesc(
            target_sparsity_by_scope={
                TargetScope(patterns=[".*nonexist.*"], validate=False): 0.3,
                TargetScope(names=["ThreeLinearModel/Linear[linear1]/linear_0"]): 0.3,
            },
            ignored_scope=IgnoredScope(),
            ref_target_sparsity_by_node_name={
                "ThreeLinearModel/Linear[linear1]/linear_0": 0.3,
            },
        ),
        TargetSparsityByNodeTestDesc(
            target_sparsity_by_scope={TargetScope(patterns=[".*nonexist.*"]): 0.3},
            ignored_scope=IgnoredScope(),
            raised_error_message="not found in the graph",
        ),
        TargetSparsityByNodeTestDesc(
            target_sparsity_by_scope={
                TargetScope(patterns=[".*linear2.*"]): 0.3,
                TargetScope(types=["embedding"]): 0.3,  # Embedding is not supported
            },
            ignored_scope=IgnoredScope(patterns=[".*linear2.*"]),
            raised_error_message="No layers to conduct activation sparsification",
        ),
        TargetSparsityByNodeTestDesc(
            target_sparsity_by_scope={
                TargetScope(names=["ThreeLinearModel/Linear[linear1]/linear_0"]): 0.3,
                TargetScope(patterns=[".*linear1.*"]): 0.4,
            },
            ignored_scope=IgnoredScope(),
            raised_error_message="matched by multiple items",
        ),
    ],
)
def test_get_target_sparsity_by_node(desc: TargetSparsityByNodeTestDesc):
    model = wrap_model(
        ThreeLinearModel(),
        example_input=torch.ones((2, 4)).long(),
        trace_parameters=True,
    )
    graph = model.nncf.get_graph()
    algo = SparsifyActivationsAlgorithm(desc.target_sparsity_by_scope, desc.ignored_scope)
    algo._set_backend_entity(model)
    if desc.raised_error_message is not None:
        with pytest.raises(nncf.ValidationError, match=desc.raised_error_message):
            algo._get_target_sparsity_by_node(graph)
    else:
        target_sparsity_by_node = algo._get_target_sparsity_by_node(graph)
        target_sparsity_by_node_name = {node.node_name: sparsity for node, sparsity in target_sparsity_by_node.items()}
        assert sorted(target_sparsity_by_node_name.items()) == sorted(desc.ref_target_sparsity_by_node_name.items())
