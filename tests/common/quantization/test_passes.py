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

from enum import Enum

import pytest

from nncf.quantization.passes import remove_dropout_nodes_inplace
from tests.post_training.test_templates.models import NNCFGraphDropoutRemovingCase
from tests.torch.test_compressed_graph import check_graph

REF_DIR = "passes/dropout_removed"


class TestModes(Enum):
    VALID = "valid"
    WRONG_TENSOR_SHAPE = "wrong_dropout_node"
    WRONG_PARALLEL_EDGES = "wrong_parallel_edges"


@pytest.mark.parametrize("mode", [TestModes.VALID, TestModes.WRONG_TENSOR_SHAPE, TestModes.WRONG_PARALLEL_EDGES])
def test_remove_dropout_nodes_inplace(mode: TestModes):
    dot_reference_path = "dropout_synthetic_model.dot"
    dropout_metatype = "DROPOUT_METATYPE"
    kwargs = {}
    if mode != TestModes.VALID:
        kwargs.update({mode.value: True})

    nncf_graph = NNCFGraphDropoutRemovingCase(dropout_metatype, **kwargs).nncf_graph
    if mode != TestModes.VALID:
        with pytest.raises(AssertionError):
            remove_dropout_nodes_inplace(nncf_graph, [dropout_metatype])
        return

    remove_dropout_nodes_inplace(nncf_graph, [dropout_metatype])
    check_graph(nncf_graph, dot_reference_path, REF_DIR)
