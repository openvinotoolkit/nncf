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
import shutil
from pathlib import Path

from nncf.common.quantization.quantizer_propagation.graph import QuantizerPropagationStateGraph
from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver


class QuantizerPropagationVisualizer:
    """
    An object performing visualization of the quantizer propagation algorithm's state into a chosen directory.
    """

    def __init__(self, dump_dir: str):
        self.dump_dir = Path(dump_dir)
        if self.dump_dir.exists():
            shutil.rmtree(str(self.dump_dir))

    def visualize_quantizer_propagation(
        self, prop_solver: QuantizerPropagationSolver, prop_graph: QuantizerPropagationStateGraph, iteration: str
    ) -> None:
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        fname = "quant_prop_iter_{}.dot".format(iteration)
        prop_solver.debug_visualize(prop_graph, str(self.dump_dir / Path(fname)))
