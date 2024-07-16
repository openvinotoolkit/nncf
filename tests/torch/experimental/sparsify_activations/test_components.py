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
from typing import List

import pytest
import torch

import nncf
import nncf.experimental
import nncf.experimental.torch.sparsify_activations
from nncf.experimental.torch.sparsify_activations.torch_backend import ActivationsSparsifier
from nncf.experimental.torch.sparsify_activations.torch_backend import PTSparsifyActivationsAlgoBackend
from nncf.torch.model_creation import wrap_model
from nncf.torch.nncf_network import NNCFNetwork
from tests.torch.experimental.sparsify_activations.helpers import ThreeLinearModel


@dataclass
class SparsifierForwardTestDesc:
    target_sparsity: float
    alpha: float
    input_batches: List[torch.Tensor]
    ref_running_thresholds: List[torch.Tensor]
    ref_outputs: List[torch.Tensor]


sparsifier_forward_test_descs = {
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
    @pytest.mark.parametrize("desc", sparsifier_forward_test_descs.values(), ids=sparsifier_forward_test_descs.keys())
    def test_forward(self, use_cuda: bool, desc: SparsifierForwardTestDesc):
        if use_cuda and not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        device = torch.device("cuda" if use_cuda else "cpu")
        sparsifier = ActivationsSparsifier(desc.target_sparsity, desc.alpha).to(device)
        sparsifier.freeze(False)

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

        sparsifier.freeze()
        with torch.no_grad():
            batch = desc.input_batches[-1]
            output = sparsifier(batch.to(device))
        assert sparsifier.num_batches_tracked == len(desc.input_batches)
        torch.testing.assert_close(
            sparsifier.running_threshold, desc.ref_running_thresholds[-1], rtol=1e-4, atol=1e-4, check_device=False
        )
        torch.testing.assert_close(output, desc.ref_outputs[-1], rtol=1e-4, atol=1e-4, check_device=False)


class TestPTSparsifyActivationsAlgoBackend:
    def test_get_sparsifiers(self):
        model = ThreeLinearModel()
        dataset = nncf.Dataset(torch.randint(0, 30, (3, 2, 8)))
        sparse_model = nncf.experimental.torch.sparsify_activations.sparsify_activations(
            model, dataset, target_sparsity_by_scope={"{re}.*": 0.5}
        )
        backend = PTSparsifyActivationsAlgoBackend()
        sparsifiers = backend.get_sparsifiers(sparse_model)
        assert len(sparsifiers) == 3

    @pytest.mark.parametrize("compress_weights", [False, True])
    def test_insert_sparsifiers(self, compress_weights: bool):
        model, _ = self.create_model_and_dataset(compress_weights=compress_weights)
        graph = model.nncf.get_graph()
        nodes = graph.get_nodes_by_metatypes(PTSparsifyActivationsAlgoBackend.SUPPORTED_METATYPES)
        backend = PTSparsifyActivationsAlgoBackend()
        model_with_sparsifiers = backend.insert_sparsifiers(model, graph, {node: 0.5 for node in nodes})
        assert len(backend.get_sparsifiers(model_with_sparsifiers)) == len(nodes)

    def test_calibrate_sparsifiers(self, mocker):
        model, dataset = self.create_model_and_dataset()
        graph = model.nncf.get_graph()
        backend = PTSparsifyActivationsAlgoBackend()
        mock_sparsifier = ActivationsSparsifier(0.5, 0.1)
        mock_sparsifier.freeze(True)
        num_model_forward_calls = 0

        def model_forward_pre_hook(model: NNCFNetwork, args):
            nonlocal num_model_forward_calls
            num_model_forward_calls += 1
            assert model.training is False

        model.register_forward_pre_hook(model_forward_pre_hook)

        with mocker.patch.object(backend, "get_sparsifiers", return_value=[mock_sparsifier]):
            backend.calibrate_sparsifiers(model, graph, dataset)
            assert mock_sparsifier._freeze is False
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
