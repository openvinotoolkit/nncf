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
from nncf.common.graph import NNCFNodeName


class PrunedLayerInfoBase:
    def __init__(self, node_name: NNCFNodeName, node_id: int, is_depthwise: bool):
        self.node_name = node_name
        self.nncf_node_id = node_id
        self.is_depthwise = is_depthwise
