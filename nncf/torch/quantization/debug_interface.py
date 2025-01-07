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
from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import networkx as nx
import numpy as np
import torch

import nncf
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.insertion_point_graph import InsertionPointGraphNodeType
from nncf.common.logging import nncf_logger
from nncf.common.utils.dot_file_rw import write_dot_graph
from nncf.common.utils.os import safe_open
from nncf.torch.debug import CallCountTracker
from nncf.torch.debug import DebugInterface
from nncf.torch.quantization.layers import QUANTIZATION_MODULES

if TYPE_CHECKING:
    from nncf.torch.nncf_network import NNCFNetwork


class QuantizationDebugInterface(DebugInterface):
    QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME = "quantized_modules"
    ACTIVATION_QUANTIZERS_TRACKER_NAME = "activation_quantizers"

    def __init__(self):
        self.call_trackers = {
            self.QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME: CallCountTracker(
                QuantizationDebugInterface.QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME
            ),
            self.ACTIVATION_QUANTIZERS_TRACKER_NAME: CallCountTracker(
                QuantizationDebugInterface.ACTIVATION_QUANTIZERS_TRACKER_NAME
            ),
        }
        self.graph_size = 0

        from nncf.common.utils.debug import DEBUG_LOG_DIR

        self.dump_dir = Path(DEBUG_LOG_DIR) / Path("debug_dumps")
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.scale_dump_dir = self.dump_dir / Path("scale")
        if self.scale_dump_dir.exists():
            shutil.rmtree(str(self.scale_dump_dir))
        self.scale_dump_dir.mkdir(parents=True, exist_ok=True)
        self.forward_call_count = 0
        self._strict_forward = False

    def init_actual(self, owner_model: NNCFNetwork):
        from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType

        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        quantizers_in_nncf_modules = owner_model.nncf.get_modules_in_nncf_modules_by_type(quantization_types)
        nncf_module_quantizations_id_list: List[str] = [str(scope) for scope in quantizers_in_nncf_modules]

        activation_quantizer_id_list: List[str] = owner_model.nncf.get_compression_modules_by_type(
            ExtraCompressionModuleType.EXTERNAL_QUANTIZER
        ).keys()
        self.call_trackers[self.QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME].init_with_key_list(
            nncf_module_quantizations_id_list
        )
        self.call_trackers[self.ACTIVATION_QUANTIZERS_TRACKER_NAME].init_with_key_list(activation_quantizer_id_list)
        self._strict_forward = True

    def pre_forward_actions(self, module: NNCFNetwork):
        self.reset_counters()

    def post_forward_actions(self, module: NNCFNetwork):
        self.register_forward_call()

        ctx = module.nncf.get_tracing_context()
        self.set_graph_size(ctx.graph.get_nodes_count())

        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        nncf_module_quantizations = module.nncf.get_modules_in_nncf_modules_by_type(quantization_types)

        for qm_scope, qm_module in nncf_module_quantizations.items():
            # Important - this will not work for DataParallel since it copies the
            # entire parent module for each thread and the `call_count` attributes
            # are incremented for thread local copies of `qm_module`, which are not
            # the same as the primary copies of `qm_module` iterated over at this point
            self.register_quantizer_module_call(str(qm_scope), qm_module.call_count)
            self.dump_scale(qm_module.get_trainable_params(), str(qm_scope))
            qm_module.reset_call_counter()
        self.print_call_stats()

        call_dict = ctx.get_node_call_counter_dict()
        total_calls = sum(call_dict.values())
        nncf_logger.debug(f"{total_calls} nodes called out of total {ctx.graph.get_nodes_count()}")
        if self._strict_forward:
            for tracker in self.call_trackers.values():
                if tracker.get_never_called_keys():
                    # This will always trigger for DataParallel - disregard or disable debug mode
                    # for DataParallel runs
                    raise nncf.InternalError(
                        f"{tracker.name} has never called modules: {tracker.get_never_called_keys()}!"
                    )

    def dump_scale(self, quantizer_scale_params: Dict[str, torch.Tensor], quantizer_name: str):
        import re

        quantizer_normalized_name = re.sub(r"[^\w\-_\. ]", "_", quantizer_name)
        for scale_param_name, scale_param in quantizer_scale_params.items():
            fname = "{}_{}.txt".format(quantizer_normalized_name, scale_param_name)
            with safe_open(self.scale_dump_dir / fname, "ab") as file:
                np.savetxt(file, scale_param.cpu().numpy().flatten())

    def reset_counters(self):
        for tracker in self.call_trackers.values():
            tracker.reset()

    def register_quantizer_module_call(self, key, counts=None):
        self.call_trackers[self.QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME].register_call(key, counts)

    def register_activation_quantize_call(self, key: str):
        self.call_trackers[self.ACTIVATION_QUANTIZERS_TRACKER_NAME].register_call(key)

    def print_call_stats(self):
        nncf_logger.debug(f" Graph size: {self.graph_size} nodes")
        for tracker in self.call_trackers.values():
            msg = f" {tracker.name} tracker:"
            msg += f" {tracker.get_total_call_count()} total calls;"

            never_called = tracker.get_never_called_keys()
            if never_called:
                msg += f" {len(never_called)} entries never called;"

            overcalled = tracker.get_overcalled_keys_with_call_counts()
            if overcalled:
                msg += f" {len(overcalled)} entries called more than once;"
            nncf_logger.debug(msg)

    def set_graph_size(self, new_size):
        if new_size != self.graph_size:
            nncf_logger.debug("\n")
            nncf_logger.debug(
                f"Warning - graph size has changed from {self.graph_size} to {new_size} since last forward"
            )
        self.graph_size = new_size

    def register_forward_call(self):
        self.forward_call_count += 1

    def visualize_insertion_point_graph(self, insertion_point_graph: InsertionPointGraph):
        out_graph = nx.MultiDiGraph()
        for node_key, node in insertion_point_graph.nodes.items():
            if node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] in [
                InsertionPointGraphNodeType.PRE_HOOK,
                InsertionPointGraphNodeType.POST_HOOK,
            ]:
                target_point_data = node[InsertionPointGraph.INSERTION_POINT_NODE_ATTR]
                label = "TP: {}".format(str(target_point_data))
                out_graph.add_node(node_key, label=label, color="red")
            elif node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR:
                out_graph.add_node(node_key)
            else:
                raise nncf.InternalError("Invalid InsertionPointGraph node!")
        for u, v in insertion_point_graph.edges:
            out_graph.add_edge(u, v)

        for node_key, node in insertion_point_graph.nodes.items():
            if node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR:
                for ip_node_key in node[InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR]:
                    out_graph.add_edge(node_key, ip_node_key, style="dashed", headport="e", tailport="e")

        write_dot_graph(out_graph, self.dump_dir / Path("insertion_point_graph.dot"))
