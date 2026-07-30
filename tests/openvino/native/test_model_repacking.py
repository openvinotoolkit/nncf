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

import openvino as ov

import nncf
from tests.openvino.native.models import ModelForRepack


def get_constant_element_types(model: ov.Model) -> list[ov.Type]:
    """Collects element types of all Constant nodes in the model."""
    types = []
    for op in model.get_ops():
        if op.get_type_name() == "Constant":
            types.append(op.get_element_type())
    return types


def test_repack_weights_produces_u3_and_u2():
    model = ModelForRepack().ov_model
    repacked_model = nncf.repack_weights(model)

    element_types = get_constant_element_types(repacked_model)
    assert ov.Type.u3 in element_types, "Expected u3 constant after repacking"
    assert ov.Type.u2 in element_types, "Expected u2 constant after repacking"
