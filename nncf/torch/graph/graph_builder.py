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

from typing import Any
from typing import Callable
from typing import List
from typing import Optional

import torch

from nncf.torch.dynamic_graph.graph_tracer import GraphTracer
from nncf.torch.dynamic_graph.graph_tracer import ModelInputInfo
from nncf.torch.graph.graph import PTNNCFGraph


class GraphBuilder:
    def __init__(self, custom_forward_fn: Callable[[torch.nn.Module], Any]):
        self.custom_forward_fn = custom_forward_fn

    def build_graph(self, model: torch.nn.Module, context_to_use: Optional['TracingContext'] = None,
                    as_eval: bool = False,
                    input_infos: List[ModelInputInfo] = None) -> PTNNCFGraph:
        tracer = GraphTracer(self.custom_forward_fn)
        traced_graph = tracer.trace_graph(model, context_to_use, as_eval)
        return PTNNCFGraph.from_dynamic_graph(traced_graph, input_infos=input_infos)
