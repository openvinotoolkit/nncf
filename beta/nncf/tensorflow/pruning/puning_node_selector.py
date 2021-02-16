"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from beta.nncf.tensorflow.graph.graph import TFNNCFNode
from beta.nncf.tensorflow.graph.graph import tf_get_layer_identifier
from beta.nncf.tensorflow.pruning.utils import tf_is_conv_with_downsampling
from beta.nncf.tensorflow.pruning.utils import tf_is_depthwise_conv
from nncf.common.pruning.pruning_node_selector import PruningNodeSelector


class TFPruningNodeSelector(PruningNodeSelector):

    def _get_module_identifier(self, node: TFNNCFNode) -> str:
        return tf_get_layer_identifier(node)

    def _is_depthwise_conv(self, node: TFNNCFNode) -> bool:
        return tf_is_depthwise_conv(node)

    def _is_conv_with_downsampling(self, node: TFNNCFNode) -> bool:
        return tf_is_conv_with_downsampling(node)
