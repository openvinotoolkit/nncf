# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import networkx as nx
import torch
from tests.torch2.utils import compare_with_reference_file
from tests.torch2.utils import to_comparable_nx_graph

import nncf
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import build_nncf_graph
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch.helpers import BasicConvTestModel

REF_DIR = TEST_ROOT / "torch2" / "data" / "function_hook" / "pruning_and_quantization"


def test_prune_ptq_model(regen_ref_data: bool):
    model = BasicConvTestModel()
    example_input = torch.ones(model.INPUT_SIZE)

    pruned = nncf.prune(
        model,
        nncf.PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL,
        ratio=0.5,
        examples_inputs=example_input,
    )

    calibrated_dataset = nncf.Dataset([example_input])
    compressed_model = nncf.quantize(pruned, calibration_dataset=calibrated_dataset)

    nncf_graph = build_nncf_graph(compressed_model, example_input)
    nx_graph = to_comparable_nx_graph(nncf_graph)
    dot_nncf_graph = nx.nx_pydot.to_pydot(nx_graph)
    ref_file = REF_DIR / "prune_ptq_model.dot"
    compare_with_reference_file(str(dot_nncf_graph), ref_file, regen_ref_data)
