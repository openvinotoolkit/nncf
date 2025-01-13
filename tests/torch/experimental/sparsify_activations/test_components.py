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
from typing import List

import pytest
import torch

import nncf
import nncf.experimental
import nncf.experimental.torch.sparsify_activations
from nncf.experimental.torch.sparsify_activations.target_scope import TargetScope
from nncf.experimental.torch.sparsify_activations.target_scope import get_target_node_names_from_target_scope
from nncf.experimental.torch.sparsify_activations.torch_backend import ActivationsSparsifier
from nncf.experimental.torch.sparsify_activations.torch_backend import PTSparsifyActivationsAlgoBackend
from nncf.torch.model_creation import wrap_model
from nncf.torch.nncf_network import NNCFNetwork
from tests.common.test_ignored_scope import CONV_TYPE
from tests.common.test_ignored_scope import IGNORED_SCOPES_TEST_DATA
from tests.common.test_ignored_scope import LINEAR_TYPE
from tests.common.test_ignored_scope import WRONG_IGNORED_SCOPES_TEST_DATA
from tests.common.test_ignored_scope import NNCFGraphToTestIgnoredScope
from tests.torch.experimental.sparsify_activations.helpers import ThreeLinearModel
from tests.torch.experimental.sparsify_activations.helpers import convert_ignored_scope_to_target_scope


@dataclass
class SparsifierForwardTestDesc:
    target_sparsity: float
    alpha: float
    input_batches: List[torch.Tensor]
    ref_running_thresholds: List[torch.Tensor]
    ref_outputs: List[torch.Tensor]


sparsifier_forward_during_calibration_test_descs = {
    "fp16": SparsifierForwardTestDesc(
        target_sparsity=0.4,
        alpha=0.2,
        input_batches=[
            torch.tensor([1.0, 3.0, 2.0, 4.0], dtype=torch.float16),
            torch.tensor([4.0, 5.0, 4.5, -3.0], dtype=torch.float16),
        ],
        ref_running_thresholds=[
            torch.tensor(2.1992, dtype=torch.float16),
            torch.tensor(3.2559, dtype=torch.float16),
        ],
        ref_outputs=[
            torch.tensor([0.0, 3.0, 0.0, 4.0], dtype=torch.float16),
            torch.tensor([4.0, 5.0, 4.5, 0.0], dtype=torch.float16),
        ],
    ),
    "fp32": SparsifierForwardTestDesc(
        target_sparsity=0.8,
        alpha=0.1,
        input_batches=[
            torch.tensor([-1.0, 1.0, 2.5]),
            torch.tensor([1.0, 2.0, 0.0]),
            torch.tensor([2.0, 0.0, 3.0]),
        ],
        ref_running_thresholds=[
            torch.tensor(1.9000),
            torch.tensor(1.7421),
            torch.tensor(2.0587),
        ],
        ref_outputs=[
            torch.tensor([0.0, 0.0, 2.5]),
            torch.tensor([0.0, 2.0, 0.0]),
            torch.tensor([0.0, 0.0, 3.0]),
        ],
    ),
    "varying_shape": SparsifierForwardTestDesc(
        target_sparsity=0.6,
        alpha=0.5,
        input_batches=[
            torch.tensor([1.0, 2.0, 7.0]),
            torch.tensor([[1.0, 2.0], [7.0, -3.0]]),
            torch.tensor([[[1.0], [5.5], [8.5], [-3.0], [2.5]]]),
        ],
        ref_running_thresholds=[
            torch.tensor(3.0000),
            torch.tensor(2.8667),
            torch.tensor(3.5143),
        ],
        ref_outputs=[
            torch.tensor([0.0, 0.0, 7.0]),
            torch.tensor([[0.0, 0.0], [7.0, -3.0]]),
            torch.tensor([[[0.0], [5.5], [8.5], [0.0], [0.0]]]),
        ],
    ),
}


class TestActivationsSparsifier:
    @pytest.fixture(autouse=True)
    def setup(self, use_cuda: bool):
        if use_cuda and not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        self.device = torch.device("cuda" if use_cuda else "cpu")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_forward_before_calibration(self, use_cuda: bool, dtype: torch.dtype):
        device = self.device
        input_tensor = torch.rand([3, 3], device=device, dtype=dtype)
        sparsifier = ActivationsSparsifier(target_sparsity=0.9).to(device)
        assert sparsifier.freeze is True
        assert not sparsifier.num_batches_tracked.is_nonzero()
        assert sparsifier.running_threshold.isneginf()
        output_tensor = sparsifier(input_tensor)
        # The output tensor is a new tensor
        assert not output_tensor.is_set_to(input_tensor)
        # Before calibration, the sparsifier does not change the input
        torch.testing.assert_close(output_tensor, input_tensor, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "desc",
        sparsifier_forward_during_calibration_test_descs.values(),
        ids=sparsifier_forward_during_calibration_test_descs.keys(),
    )
    def test_forward_during_calibration(self, use_cuda: bool, desc: SparsifierForwardTestDesc):
        device = self.device
        sparsifier = ActivationsSparsifier(desc.target_sparsity, desc.alpha).to(device)
        sparsifier.freeze = False
        running_thresholds = []
        outputs = []
        with torch.no_grad():
            for batch in desc.input_batches:
                output = sparsifier(batch.to(device))
                running_thresholds.append(sparsifier.running_threshold)
                outputs.append(output)
        assert sparsifier.num_batches_tracked == len(desc.input_batches)
        assert len(running_thresholds) == len(desc.ref_running_thresholds)
        for threshold, ref_threshold in zip(running_thresholds, desc.ref_running_thresholds):
            assert threshold.device.type == device.type
            torch.testing.assert_close(threshold, ref_threshold, rtol=1e-4, atol=1e-4, check_device=False)
        assert len(outputs) == len(desc.ref_outputs)
        for output, ref_output in zip(outputs, desc.ref_outputs):
            assert output.device.type == device.type
            torch.testing.assert_close(output, ref_output, rtol=1e-4, atol=1e-4, check_device=False)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_forward_after_calibration(self, use_cuda: bool, dtype: torch.dtype):
        device = self.device
        sparsifier = ActivationsSparsifier(target_sparsity=0.9).to(device)
        sparsifier.running_threshold.fill_(0.1)
        sparsifier.num_batches_tracked.fill_(100)

        for _ in range(2):
            # The sparsifier does not change in the following forwards
            input_tensor = torch.rand([2, 10], device=device, dtype=dtype)
            ref_output = torch.where(input_tensor.abs() <= 0.1, 0.0, input_tensor)
            output_tensor = sparsifier(ref_output)
            assert sparsifier.num_batches_tracked == 100
            torch.testing.assert_close(
                sparsifier.running_threshold, torch.tensor(0.1, device=device), rtol=1e-4, atol=1e-4
            )
            torch.testing.assert_close(output_tensor, ref_output, rtol=1e-4, atol=1e-4)


class TestPTSparsifyActivationsAlgoBackend:
    def test_get_sparsifiers(self):
        model, dataset = self.create_model_and_dataset()
        sparse_model = nncf.experimental.torch.sparsify_activations.sparsify_activations(
            model, dataset, target_sparsity_by_scope={TargetScope(patterns=[".*"]): 0.5}
        )
        backend = PTSparsifyActivationsAlgoBackend()
        sparsifiers = backend.get_sparsifiers(sparse_model)
        assert len(sparsifiers) == 3

    @pytest.mark.parametrize("compress_weights", [False, True])
    def test_insert_sparsifiers(self, compress_weights: bool):
        model, dataset = self.create_model_and_dataset(compress_weights=compress_weights)
        example_input = next(iter(dataset.get_inference_data()))
        ref_output = model(example_input)

        graph = model.nncf.get_graph()
        nodes = graph.get_nodes_by_metatypes(PTSparsifyActivationsAlgoBackend.SUPPORTED_METATYPES)
        backend = PTSparsifyActivationsAlgoBackend()
        model_with_sparsifiers = backend.insert_sparsifiers(model, graph, {node: 0.9 for node in nodes})
        assert len(backend.get_sparsifiers(model_with_sparsifiers)) == len(nodes)

        output = model_with_sparsifiers(example_input)
        torch.testing.assert_close(
            output, ref_output, rtol=1e-4, atol=1e-4
        )  # At this time the sparsifers do not change the output

    def test_calibrate_sparsifiers(self, mocker):
        model, dataset = self.create_model_and_dataset()
        graph = model.nncf.get_graph()
        backend = PTSparsifyActivationsAlgoBackend()
        mock_sparsifier = ActivationsSparsifier(0.5, 0.1)
        mock_sparsifier.freeze = True
        num_model_forward_calls = 0

        def model_forward_pre_hook(model: NNCFNetwork, args):
            nonlocal num_model_forward_calls
            num_model_forward_calls += 1
            assert model.training is False

        model.register_forward_pre_hook(model_forward_pre_hook)

        with mocker.patch.object(backend, "get_sparsifiers", return_value=[mock_sparsifier]):
            backend.calibrate_sparsifiers(model, graph, dataset)
            assert mock_sparsifier.freeze is True
            assert num_model_forward_calls == dataset.get_length()

    def create_model_and_dataset(self, compress_weights: bool = False):
        model = ThreeLinearModel()
        dataset = nncf.Dataset(torch.randint(0, 30, (3, 2, 8)))
        if compress_weights:
            model = nncf.compress_weights(
                model,
                mode=nncf.CompressWeightsMode.INT8_SYM,
                dataset=dataset,
            )
        else:
            model = wrap_model(
                model,
                example_input=next(iter(dataset.get_inference_data())),
                trace_parameters=True,
            )
        return model, dataset


class TestTargetScope:
    SAME_HASH_PAIRS = [
        (TargetScope(), TargetScope()),
        (
            TargetScope(
                names=["node_1", "node_2"],
                patterns=["node\\d", "layer\\d"],
                types=["Conv", "MatMul"],
                subgraphs=[
                    nncf.Subgraph(inputs=["node_1", "node_2"], outputs=["node_3", "node_4"]),
                    nncf.Subgraph(inputs=["layer_1", "layer_2"], outputs=["layer_3", "layer_4", "layer_5"]),
                ],
            ),
            TargetScope(
                names=["node_2", "node_1"],
                patterns=["layer\\d", "node\\d"],
                types=["MatMul", "Conv"],
                subgraphs=[
                    nncf.Subgraph(inputs=["layer_2", "layer_1"], outputs=["layer_5", "layer_4", "layer_3"]),
                    nncf.Subgraph(inputs=["node_2", "node_1"], outputs=["node_4", "node_3"]),
                ],
            ),
        ),
    ]

    DIFFERENT_HASH_PAIRS = [
        (TargetScope(), TargetScope(types=["Conv"])),
        (
            TargetScope(names=["node_1"]),
            TargetScope(names=["node_1"], patterns=["layer\\d"]),
        ),
        (
            TargetScope(subgraphs=[nncf.Subgraph(inputs=["node_1"], outputs=["node_2"])]),
            TargetScope(subgraphs=[nncf.Subgraph(inputs=["node_1"], outputs=["node_3"])]),
        ),
    ]

    TARGET_SCOPE_MATCH_DATA = [
        (convert_ignored_scope_to_target_scope(ignored_scope), ref_ignored_names)
        for ignored_scope, ref_ignored_names in IGNORED_SCOPES_TEST_DATA
    ]
    WRONG_TARGET_SCOPE_MATCH_DATA = list(map(convert_ignored_scope_to_target_scope, WRONG_IGNORED_SCOPES_TEST_DATA))

    @pytest.mark.parametrize("target_scope1,target_scope2", SAME_HASH_PAIRS)
    def test_same_hash(self, target_scope1: TargetScope, target_scope2: TargetScope):
        assert hash(target_scope1) == hash(target_scope2)

    @pytest.mark.parametrize("target_scope1,target_scope2", DIFFERENT_HASH_PAIRS)
    def test_different_hash(self, target_scope1: TargetScope, target_scope2: TargetScope):
        assert hash(target_scope1) != hash(target_scope2)

    @pytest.mark.parametrize("target_scope,ref_target_names", TARGET_SCOPE_MATCH_DATA)
    def test_get_target_node_names_from_target_scope(self, target_scope: TargetScope, ref_target_names: List[str]):
        nncf_graph = NNCFGraphToTestIgnoredScope(CONV_TYPE, LINEAR_TYPE).nncf_graph
        target_names = get_target_node_names_from_target_scope(target_scope, nncf_graph)
        assert sorted(target_names) == sorted(ref_target_names)

    @pytest.mark.parametrize("target_scope", WRONG_TARGET_SCOPE_MATCH_DATA)
    def test_wrong_target_scope(self, target_scope: TargetScope):
        nncf_graph = NNCFGraphToTestIgnoredScope(CONV_TYPE, LINEAR_TYPE).nncf_graph
        with pytest.raises(nncf.ValidationError):
            get_target_node_names_from_target_scope(target_scope, nncf_graph)
