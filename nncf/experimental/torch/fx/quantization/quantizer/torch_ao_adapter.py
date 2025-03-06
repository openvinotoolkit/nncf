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


from collections import defaultdict
from typing import Dict, List, Tuple, Union

import torch
import torch.fx
from torch.ao.quantization.quantizer import Quantizer as TorchAOQuantizer
from torch.ao.quantization.quantizer.quantizer import QuantizationSpec
from torch.ao.quantization.quantizer.quantizer import QuantizationSpecBase
from torch.ao.quantization.quantizer.quantizer import SharedQuantizationSpec

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.logging import nncf_logger
from nncf.common.quantization.quantizer_setup import ActivationQuantizationInsertionPoint
from nncf.common.quantization.quantizer_setup import QuantizationPointBase
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizationPoint
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.quantizer_setup import WeightQuantizationInsertionPoint
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.experimental.quantization.quantizer import Quantizer
from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter

EdgeOrNode = Union[Tuple[torch.fx.Node, torch.fx.Node]]


class TorchAOQuantizerAdapter(Quantizer):
    """
    Implementation of the NNCF Quantizer interface for any given torch.ao quantizer.
    """

    def __init__(self, quantizer: TorchAOQuantizer):
        self._quantizer = quantizer

    def transform_prior_quantization(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        return self._quantizer.transform_for_annotation(model)

    def get_quantization_setup(self, model: torch.fx.GraphModule, nncf_graph: NNCFGraph) -> SingleConfigQuantizerSetup:
        # Save model and nodes meta before the annotation
        original_meta = model.meta.copy()
        node_name_vs_meta = {}
        with torch.no_grad():
            for node in model.graph.nodes:
                node_name_vs_meta[node.name] = node.meta.copy()

        annotated_model = self._quantizer.annotate(model)
        self._quantizer.validate(annotated_model)
        quantizer_setup = self.get_quantizer_config_from_annotated_model(annotated_model)

        # Recover original meta
        model.meta = original_meta
        for node in model.graph.nodes:
            node.meta = node_name_vs_meta[node.name]

        return quantizer_setup

    @staticmethod
    def _get_quantization_points(
        from_node: torch.fx.Node,
        to_nodes: List[torch.fx.Node],
        annotated_model: torch.fx.GraphModule,
        qconfig: QuantizerConfig,
    ) -> List[QuantizationPointBase]:
        to_n = to_nodes[0]
        if from_node.op == "get_attr":
            _, metatype = GraphConverter.get_node_type_and_metatype(to_n, annotated_model)
            # Check that the constant is placed on the actual weight port, as it is possible for
            # activations to be a constant as well.
            if TorchAOQuantizerAdapter._get_node_args(to_n).index(from_node) in metatype.weight_port_ids:
                qip = WeightQuantizationInsertionPoint(to_n.name)
                return [SingleConfigQuantizationPoint(qip, qconfig, [x.name for x in to_nodes])]

        if len(from_node.users) == len(to_nodes):
            qip = ActivationQuantizationInsertionPoint(from_node.name)
            return [SingleConfigQuantizationPoint(qip, qconfig, [x.name for x in to_nodes])]

        qps = []
        for to_n_ in to_nodes:
            input_port_id = to_n_.args.index(from_node)
            qip = ActivationQuantizationInsertionPoint(to_n_.name, input_port_id)
            qp = SingleConfigQuantizationPoint(qip, qconfig, [to_n_.name])
            qps.append(qp)
        return qps

    @staticmethod
    def _get_node_args(node: torch.fx.Node):
        if node.target == torch.ops.aten.cat.default:
            return node.args[0]
        return node.args

    @staticmethod
    def get_quantizer_config_from_annotated_model(annotated_model: torch.fx.GraphModule) -> SingleConfigQuantizerSetup:
        edge_or_node_to_qspec = _get_edge_or_node_to_qspec(annotated_model)

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
                    msg = f"Unknown qscheme: {qspec.qscheme}"
                    raise nncf.InternalError(msg)
                signed = qspec.dtype is torch.int8
                mode = (
                    QuantizationMode.SYMMETRIC
                    if qspec.qscheme in [torch.per_channel_symmetric, torch.per_tensor_symmetric]
                    else QuantizationMode.ASYMMETRIC
                )
                qconfig = QuantizerConfig(mode=mode, signedness_to_force=signed, per_channel=per_channel)

                qps = TorchAOQuantizerAdapter._get_quantization_points(from_n, to_nodes, annotated_model, qconfig)
                for qp in qps:
                    q_setup.add_independent_quantization_point(qp)

            elif isinstance(qspec, SharedQuantizationSpec):
                # TODO(dlyakhov): Support SharedQuantizationSpec
                nncf_logger.warning(
                    f"SharedQuantizationSpec is not supported yet; edges {from_n} -> {to_nodes} won't be quantized."
                )
            else:
                msg = f"Unknown torch.ao quantization spec: {qspec}"
                raise nncf.InternalError(msg)

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
