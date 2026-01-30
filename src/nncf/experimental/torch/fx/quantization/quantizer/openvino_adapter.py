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

from typing import Any

import torch.fx

from nncf.common.graph.graph import NNCFGraph
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.experimental.quantization.quantizer import Quantizer
from nncf.experimental.torch.fx.quantization.quantizer.openvino_quantizer import OpenVINOQuantizer
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters


class OpenVINOQuantizerAdapter(Quantizer):
    """
    Implementation of the NNCF Quantizer interface for the OpenVINOQuantizer.
    """

    def __init__(self, quantizer: OpenVINOQuantizer):
        self._quantizer = quantizer

    def transform_prior_quantization(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        return self._quantizer.transform_for_annotation(model)

    def get_quantization_setup(self, model: torch.fx.GraphModule, nncf_graph: NNCFGraph) -> SingleConfigQuantizerSetup:
        return self._quantizer.get_nncf_quantization_setup(model, nncf_graph)

    def get_weight_compression_parameters(
        self,
        model: torch.fx.GraphModule,
        nncf_graph: NNCFGraph,
    ) -> tuple[
        list[WeightCompressionParameters],
        list[WeightCompressionParameters],
        list[WeightCompressionParameters],
    ]:
        return self._quantizer.get_nncf_weight_compression_parameters(model, nncf_graph)

    def get_weight_compression_config(self) -> dict[str, Any]:
        return self._quantizer.get_weights_compression_config()
