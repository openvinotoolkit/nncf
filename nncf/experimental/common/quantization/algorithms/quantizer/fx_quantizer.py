# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections import defaultdict
from copy import deepcopy
from typing import Dict, Tuple, Union

import torch
import torch.fx
from torch.ao.quantization.quantizer import Quantizer
from torch.ao.quantization.quantizer.quantizer import QuantizationSpec
from torch.ao.quantization.quantizer.quantizer import QuantizationSpecBase
from torch.ao.quantization.quantizer.quantizer import SharedQuantizationSpec

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.quantization.quantizer_setup import ActivationQuantizationInsertionPoint
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizationPoint
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.quantizer_setup import WeightQuantizationInsertionPoint
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.experimental.common.quantization.algorithms.quantizer.base_quantizer import NNCFQuantizer

EdgeOrNode = Union[Tuple[torch.fx.Node, torch.fx.Node]]


class NNCFFXQuantizer(NNCFQuantizer):
    def __init__(self, quantizer: Quantizer):
        self._quantizer = quantizer

    def get_quantization_setup(self, model: torch.fx.GraphModule, nncf_graph: NNCFGraph) -> SingleConfigQuantizerSetup:
        anotated_model = deepcopy(model)

        # self._quantizer.transform_for_annotation is called in the nncf quantize_pt2e method
        # before the nncf_graph building.
        self._quantizer.annotate(anotated_model)
        self._quantizer.validate(anotated_model)
        return self.get_quantizer_config_from_anotated_model(anotated_model)

    @staticmethod
    def get_quantizer_config_from_anotated_model(anotated_model: torch.fx.GraphModule) -> SingleConfigQuantizerSetup:
        edge_or_node_to_qspec = _get_edge_or_node_to_qspec(anotated_model)

        q_map = defaultdict(list)
        for edge, qspec in edge_or_node_to_qspec.items():
            if not isinstance(edge, tuple):
                continue
            from_n, to_n = edge
            q_map[from_n].append(to_n)

        q_setup = SingleConfigQuantizerSetup()
        for from_n, to_nodes in q_map.items():
            to_n = to_nodes[0]
            qspec = edge_or_node_to_qspec[(from_n, to_n)]
            if qspec is None:
                continue
            if isinstance(qspec, QuantizationSpec):
                if qspec.qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
                    per_channel = True
                elif qspec.qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
                    per_channel = False
                else:
                    raise nncf.InternalError(f"Unknown qscheme: {qspec.qscheme}")
                signed = qspec.dtype is torch.uint8
                mode = (
                    QuantizationMode.SYMMETRIC
                    if qspec.qscheme in [torch.per_channel_symmetric, torch.per_tensor_symmetric]
                    else QuantizationMode.ASYMMETRIC
                )
                qconfig = QuantizerConfig(mode=mode, signedness_to_force=signed, per_channel=per_channel)
                qps = []
                # If input node is a constant and placed not at activations port (0)
                if from_n.op == "get_attr" and to_n.args.index(from_n) != 0:
                    qip = WeightQuantizationInsertionPoint(to_n.name)
                    qp = SingleConfigQuantizationPoint(qip, qconfig, [x.name for x in to_nodes])
                    qps.append(qp)
                else:
                    if len(from_n.users) == len(to_nodes):
                        qip = ActivationQuantizationInsertionPoint(from_n.name)
                        qp = SingleConfigQuantizationPoint(qip, qconfig, [x.name for x in to_nodes])
                        qps.append(qp)
                    else:
                        for to_n_ in to_nodes:
                            input_port_id = to_n_.args.index(from_n)
                            qip = ActivationQuantizationInsertionPoint(to_n_.name, input_port_id)
                            qp = SingleConfigQuantizationPoint(qip, qconfig, [to_n_.name])
                            qps.append(qp)

                for qp in qps:
                    q_setup.add_independent_quantization_point(qp)

            elif isinstance(qspec, SharedQuantizationSpec):
                pass
            else:
                raise nncf.InternalError(f"Unknown torch.ao quantization spec: {qspec}")

        return q_setup


def _get_edge_or_node_to_qspec(
    model: torch.fx.GraphModule,
) -> Dict[EdgeOrNode, QuantizationSpecBase]:
    """
    Get a map from EdgeOrNode to quantization spec based on annotations on the nodes.

    :param model: torch.fx.GraphModule instance.
    :return: A map from EdgeOrNode to quantization spec based on annotations on the nodes.
    """
    edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpecBase] = {}
    for n in model.graph.nodes:
        if hasattr(n, "meta") and "quantization_annotation" in n.meta:
            qa = n.meta["quantization_annotation"]
            for input_to_n, qspec in qa.input_qspec_map.items():
                input_edge = (input_to_n, n)
                edge_or_node_to_qspec[input_edge] = qspec
            if qa.output_qspec is not None:
                output_node = n
                qspec = qa.output_qspec
                edge_or_node_to_qspec[output_node] = qspec
    return edge_or_node_to_qspec