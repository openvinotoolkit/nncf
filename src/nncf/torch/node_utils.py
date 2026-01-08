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

import nncf
import nncf.torch.graph.operator_metatypes as op
from nncf.common.graph import NNCFNode
from nncf.torch.graph.operator_metatypes import PTAddmmMetatype
from nncf.torch.graph.operator_metatypes import PTMatMulMetatype


def get_activation_channel_axis(node: NNCFNode, port_id: int) -> int:
    """
    Returns axis number of the activation tensor which correspond to its channel.

    :param node: NNCFNode instance.
    :param port_id: Port ID for input.
    :return: Channel axis number.
    """
    if node.metatype not in op.CONVOLUTION_METATYPES + op.MATMUL_METATYPES:
        msg = f"Activation channel axis retrieval from node with metatype {node.metatype} is not supported"
        raise nncf.InternalError(msg)

    if node.metatype not in [PTMatMulMetatype, PTAddmmMetatype]:
        return node.metatype.output_channel_axis

    if port_id == 0:
        # X(port:0) * W(port:1): [..., C_IN] * [... , C_IN, C_OUT]
        return -1
    if port_id == 1:
        # W(port:0) * X(port:1): [... , C_OUT, C_IN] * [... , C_IN, ...]
        return -2

    msg = f"Port id for a {node.metatype} operation is expected to be in [0, 1], {port_id} recieved"
    raise nncf.InternalError(msg)
