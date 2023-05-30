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

from nncf.openvino.graph.metatypes.openvino_metatypes import GENERAL_WEIGHT_LAYER_METATYPES
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.nncf_graph_builder import OVConstantLayerAttributes


def get_weight_channel_axis(node, weights_port_id):
    if node.metatype not in GENERAL_WEIGHT_LAYER_METATYPES:
        raise ValueError("Channel axis cannot be defined for operation without weights.")

    channel_axis = node.metatype.const_channel_axis
    if node.metatype == OVMatMulMetatype:
        assert isinstance(node.layer_attributes, OVConstantLayerAttributes)
        const_attrs = node.layer_attributes.const_attrs[weights_port_id]
        if const_attrs["transpose"]:
            channel_axis = 1 - channel_axis
    return channel_axis
