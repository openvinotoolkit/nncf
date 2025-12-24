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


from copy import deepcopy

import networkx as nx
import pytest
import torch
from torch import nn

import nncf
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import build_nncf_graph
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch.helpers import BasicConvTestModel
from tests.torch2.utils import compare_with_reference_file
from tests.torch2.utils import to_comparable_nx_graph

REF_DIR = TEST_ROOT / "torch2" / "data" / "function_hook" / "pruning_and_quantization"


@pytest.fixture(scope="session")
def prepare_model() -> nn.Module:
    model = BasicConvTestModel()
    example_input = torch.ones(model.INPUT_SIZE)

    pruned = nncf.prune(
        model,
        nncf.PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL,
        ratio=0.5,
        examples_inputs=example_input,
    )

    calibrated_dataset = nncf.Dataset([example_input])
    quantized = nncf.quantize(pruned, calibration_dataset=calibrated_dataset)
    return quantized


@pytest.fixture
def compressed_model(prepare_model: nn.Module) -> nn.Module:
    return deepcopy(prepare_model)


def test_prune_ptq_model(compressed_model: nn.Module, regen_ref_data: bool):
    example_input = torch.ones(compressed_model.INPUT_SIZE)

    nncf_graph = build_nncf_graph(compressed_model, example_input)
    nx_graph = to_comparable_nx_graph(nncf_graph)
    dot_nncf_graph = nx.nx_pydot.to_pydot(nx_graph)
    ref_file = REF_DIR / "prune_ptq_model.dot"
    compare_with_reference_file(str(dot_nncf_graph), ref_file, regen_ref_data)


def test_strip_inplace(compressed_model: nn.Module, regen_ref_data: bool):
    example_input = torch.ones(compressed_model.INPUT_SIZE)

    striped = nncf.strip(compressed_model, strip_format=nncf.StripFormat.PRUNE_IN_PLACE, do_copy=False)

    nncf_graph = build_nncf_graph(striped, example_input)
    nx_graph = to_comparable_nx_graph(nncf_graph)
    dot_nncf_graph = nx.nx_pydot.to_pydot(nx_graph)

    ref_file = REF_DIR / "strip_inplace.dot"
    compare_with_reference_file(str(dot_nncf_graph), ref_file, regen_ref_data)
