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

import pytest

import nncf
import nncf.torch.graph.operator_metatypes as op
from nncf.common.graph import NNCFNode
from nncf.torch.node_utils import get_activation_channel_axis


@pytest.mark.parametrize(
    "metatype,port_id,ref_out",
    (
        (op.PTLinearMetatype, 0, -1),
        (op.PTConv2dMetatype, 0, 1),
        (op.PTDepthwiseConv2dSubtype, 0, 1),
        (op.PTConvTranspose2dMetatype, 0, 1),
        (op.PTMatMulMetatype, 0, -1),
        (op.PTMatMulMetatype, 1, -2),
        (op.PTAddmmMetatype, 0, -1),
        (op.PTAddmmMetatype, 1, -2),
        (op.PTMatMulMetatype, 2, "error"),
        (op.PTAddMetatype, 0, "error"),
    ),
)
def test_get_activation_channel_axis(metatype, port_id, ref_out):
    node = NNCFNode({"metatype": metatype})
    if ref_out == "error":
        with pytest.raises(nncf.InternalError):
            get_activation_channel_axis(node, port_id)
    else:
        assert get_activation_channel_axis(node, port_id) == ref_out
