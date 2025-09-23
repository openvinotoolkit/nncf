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
from typing import Optional, Union

import torch.fx
from torch.ao.quantization.observer import HistogramObserver
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from torch.ao.quantization.quantizer.quantizer import EdgeOrNode
from torch.ao.quantization.quantizer.quantizer import QuantizationAnnotation as TorchAOQuantizationAnnotation
from torch.ao.quantization.quantizer.quantizer import QuantizationSpec as TorchAOQuantizationSpec
from torch.ao.quantization.quantizer.quantizer import QuantizationSpecBase as TorchAOQuantizationSpecBase
from torch.ao.quantization.quantizer.quantizer import Quantizer as TorchAOQuantizer
from torch.ao.quantization.quantizer.quantizer import SharedQuantizationSpec as TorchAOSharedQuantizationSpec

import nncf
from nncf import IgnoredScope
from nncf import ModelType
from nncf import OverflowFix
from nncf import QuantizationMode
from nncf import QuantizationPreset
from nncf import TargetDevice
from nncf.common.graph.graph import NNCFGraph
from nncf.common.logging import nncf_logger
from nncf.common.quantization.quantizer_propagation.structs import QuantizerPropagationRule
from nncf.common.quantization.quantizer_setup import QuantizationPointBase
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.structs import QuantizationScheme
from nncf.common.utils.api_marker import api
from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter
from nncf.experimental.torch.fx.node_utils import get_graph_node_by_name
from nncf.quantization.advanced_parameters import FP8QuantizationParameters
from nncf.quantization.advanced_parameters import QuantizationParameters
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.torch.model_graph_manager import get_weight_tensor_port_ids

QUANT_ANNOTATION_KEY = "quantization_annotation"


@api(canonical_alias="nncf.experimental.torch.fx.OpenVINOQuantizer")
class OpenVINOQuantizer(TorchAOQuantizer):
    """
    Implementation of the Torch AO quantizer which annotates models with quantization annotations
    optimally for the inference via OpenVINO.

    :param mode: Defines optimization mode for the algorithm. None by default.
    :param preset: A preset controls the quantization mode (symmetric and asymmetric).
        It can take the following values:
        - `performance`: Symmetric quantization of weights and activations.
        - `mixed`: Symmetric quantization of weights and asymmetric quantization of activations.
        Default value is None. In this case, `mixed` preset is used for `transformer`
        model type otherwise `performance`.
    :param target_device: A target device the specificity of which will be taken
        into account while compressing in order to obtain the best performance
        for this type of device, defaults to TargetDevice.ANY.
    :param model_type: Model type is needed to specify additional patterns
        in the model. Supported only `transformer` now.
    :param ignored_scope: An ignored scope that defined the list of model control
        flow graph nodes to be ignored during quantization.
    :param overflow_fix: This option controls whether to apply the overflow issue
        fix for the 8-bit quantization.
    :param quantize_outputs: Whether to insert additional quantizers right before
        each of the model outputs.
    :param activations_quantization_params: Quantization parameters for model
        activations.
    :param weights_quantization_params: Quantization parameters for model weights.
    :param quantizer_propagation_rule: The strategy to be used while propagating and merging quantizers.
        MERGE_ALL_IN_ONE by default.
    """

    def __init__(
        self,
        *,
        mode: Optional[QuantizationMode] = None,
        preset: Optional[QuantizationPreset] = None,
        target_device: TargetDevice = TargetDevice.ANY,
        model_type: Optional[ModelType] = None,
        ignored_scope: Optional[IgnoredScope] = None,
        overflow_fix: Optional[OverflowFix] = None,
        quantize_outputs: bool = False,
        activations_quantization_params: Optional[Union[QuantizationParameters, FP8QuantizationParameters]] = None,
        weights_quantization_params: Optional[Union[QuantizationParameters, FP8QuantizationParameters]] = None,
        quantizer_propagation_rule: QuantizerPropagationRule = QuantizerPropagationRule.MERGE_ALL_IN_ONE,
    ):
        self._min_max_algo = MinMaxQuantization(
            mode=mode,
            preset=preset,
            target_device=target_device,
            model_type=model_type,
            ignored_scope=ignored_scope,
            overflow_fix=overflow_fix,
            quantize_outputs=quantize_outputs,
            activations_quantization_params=activations_quantization_params,
            weights_quantization_params=weights_quantization_params,
            quantizer_propagation_rule=quantizer_propagation_rule,
        )

    def set_ignored_scope(
        self,
        names: Optional[list[str]] = None,
        patterns: Optional[list[str]] = None,
        types: Optional[list[str]] = None,
        subgraphs: Optional[list[tuple[list[str], list[str]]]] = None,
        validate: bool = True,
    ) -> None:
        """
        Provides an option to specify portions of model to be excluded from compression.
        The ignored scope defines model sub-graphs that should be excluded from the quantization process.

        :param names: List of ignored node names.
        :param patterns: List of regular expressions that define patterns for names of ignored nodes.
        :param types: List of ignored operation types.
        :param subgraphs: List of ignored subgraphs.
        :param validate: If set to True, then a RuntimeError will be raised if any ignored scope does not match
            in the model graph.
        """
        self._min_max_algo.set_ignored_scope(
            nncf.IgnoredScope(
                names=names or [],
                patterns=patterns or [],
                types=types or [],
                subgraphs=subgraphs or [],
                validate=validate,
            )
        )

    def get_nncf_quantization_setup(
        self, model: torch.fx.GraphModule, nncf_graph: NNCFGraph
    ) -> SingleConfigQuantizerSetup:
        self._min_max_algo._set_backend_entity(model)
        return self._min_max_algo.find_quantization_setup(model, nncf_graph)

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """
        Adds quantization annotations to the nodes in the model graph in-place.

        :param model: A torch.fx.GraphModule to annotate.
        :return: The torch.fx.GraphModule with updated annotations.
        """
        nncf_graph = GraphConverter.create_nncf_graph(model)
        quantization_setup = self.get_nncf_quantization_setup(model, nncf_graph)

        graph = model.graph
        node_vs_torch_annotation = defaultdict(TorchAOQuantizationAnnotation)

        for qp in quantization_setup.quantization_points.values():
            edge_or_node, annotation = self._get_edge_or_node_and_annotation(
                graph, nncf_graph, qp, node_vs_torch_annotation
            )
            qspec = self._get_torch_ao_qspec_from_qp(qp)
            self._fill_torch_ao_annotation(edge_or_node, qspec, annotation)

        for quantizer_ids in quantization_setup.unified_scale_groups.values():
            root_quantizer_id = self._get_unified_scales_root_quantizer_id(
                nncf_graph, quantizer_ids, quantization_setup
            )
            root_qp = quantization_setup.quantization_points[root_quantizer_id]

            if any(root_qp.qconfig != quantization_setup.quantization_points[q_id].qconfig for q_id in quantizer_ids):
                qps = [quantization_setup.quantization_points[q_id] for q_id in quantizer_ids]
                msg = (
                    "Different quantization configs are set to one unified scale group:"
                    f"{[(qp.insertion_point.__dict__, str(qp.qconfig)) for qp in qps]}"
                )
                raise nncf.InternalError(msg)

            root_target_node = get_graph_node_by_name(graph, root_qp.insertion_point.target_node_name)
            root_edge_or_node = self._get_edge_or_node(root_target_node, root_qp, nncf_graph)

            for quantizer_id in quantizer_ids:
                if quantizer_id == root_quantizer_id:
                    continue

                qspec = TorchAOSharedQuantizationSpec(root_edge_or_node)
                qp = quantization_setup.quantization_points[quantizer_id]
                edge_or_node, annotation = self._get_edge_or_node_and_annotation(
                    graph, nncf_graph, qp, node_vs_torch_annotation
                )
                self._fill_torch_ao_annotation(edge_or_node, qspec, annotation)

        for node, annotation in node_vs_torch_annotation.items():
            assert QUANT_ANNOTATION_KEY not in node.meta
            node.meta[QUANT_ANNOTATION_KEY] = annotation
        return model

    @staticmethod
    def _get_unified_scales_root_quantizer_id(
        nncf_graph: NNCFGraph, quantizer_ids: list[int], quantizer_setup: SingleConfigQuantizerSetup
    ) -> int:
        """
        Identifies the earliest quantizer node ID based on the corresponding `nncf_node.node_id`
        in the given NNCFGraph. This is required by the `_get_obs_or_fq_map` function.
        Refer to: https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/pt2e/prepare.py#L291

        :param nncf_graph: The NNCFGraph instance.
        :param quantizer_ids: The list of quantizer IDs to evaluate.
        :param quantizer_setup: The instance of SingleConfigQuantizerSetup.
        :return: The ID of the earliest quantizer node in terms of `nncf_node.node_id`.
        """
        nncf_node_quantizer_id = None
        root_quantizer_id = None
        for quantizer_id in quantizer_ids:
            target_node_name = quantizer_setup.quantization_points[quantizer_id].insertion_point.target_node_name
            nncf_node = nncf_graph.get_node_by_name(target_node_name)
            if nncf_node_quantizer_id is None or nncf_node.node_id < nncf_node_quantizer_id:
                root_quantizer_id = quantizer_id
                nncf_node_quantizer_id = nncf_node.node_id
        return root_quantizer_id

    @staticmethod
    def _get_edge_or_node_and_annotation(
        graph: torch.fx.Graph,
        nncf_graph: NNCFGraph,
        qp: QuantizationPointBase,
        node_vs_torch_annotation: dict[torch.fx.Node, TorchAOQuantizationAnnotation],
    ) -> tuple[EdgeOrNode, TorchAOQuantizationAnnotation]:
        """
        Retrieves the edge or node and its corresponding TorchAOQuantizationAnnotation based on the given graph,
        quantization point, and node-to-annotation mapping.

        :param graph: torch.fx.Graph instance.
        :param nncf_graph: NNCFGraph instance.
        :param qp: QuantizationPointBase instance.
        :param node_vs_torch_annotation: A dictionary mapping torch.fx.GraphNode objects to their respective
            TorchAOQuantizationAnnotations.
        :return: A tuple containing the EdgeOrNode and its associated TorchAOQuantizationAnnotation.
        """
        target_node = get_graph_node_by_name(graph, qp.insertion_point.target_node_name)
        annotation = node_vs_torch_annotation[target_node]
        edge_or_node = OpenVINOQuantizer._get_edge_or_node(target_node, qp, nncf_graph)
        return edge_or_node, annotation

    @staticmethod
    def _get_edge_or_node(target_node: torch.fx.Node, qp: QuantizationPointBase, nncf_graph: NNCFGraph) -> EdgeOrNode:
        """
        Returns the edge or node based on the given target node and quantization point.

        :param target_node: Target node instance.
        :param qp: QuantizationPointBase instance.
        :param graph: NNCFGraph instance.
        :return: The corresponding EdgeOrNode derived from the target node and quantization point.
        """
        ip = qp.insertion_point
        if qp.is_weight_quantization_point():
            nncf_node = nncf_graph.get_node_by_name(target_node.name)
            weights_ports_ids = get_weight_tensor_port_ids(nncf_node, nncf_graph)
            if len(weights_ports_ids) > 1:
                # TODO(dlyakhov): support quantization for nodes with several weights
                nncf_logger.warning(
                    f"Quantization of the weighted node {target_node.name}"
                    " is not yet supported by the OpenVINOQuantizer."
                    f" Only the weight on port ID {weights_ports_ids[0]} will be quantized."
                    f" Quantizable weights are located on ports: {weights_ports_ids}."
                )
            weight_node = target_node.all_input_nodes[weights_ports_ids[0]]
            return (weight_node, target_node)

        if ip.input_port_id is None:
            return target_node

        node = target_node.all_input_nodes[ip.input_port_id]
        return (node, target_node)

    @staticmethod
    def _fill_torch_ao_annotation(
        edge_or_node: EdgeOrNode,
        qspec: TorchAOQuantizationSpecBase,
        annotation_to_update: TorchAOQuantizationAnnotation,
    ) -> None:
        """
        Helper method to update the annotation_to_update based on the specified edge_or_node and qspec.

        :param edge_or_node: The target EdgeOrNode to be used for the update.
        :param qspec: An instance of TorchAOQuantizationSpecBase representing the quantization specification to apply.
        :param annotation_to_update: The annotation to update based on the edge_or_node and qspec.
        """
        if isinstance(edge_or_node, torch.fx.Node):
            annotation_to_update.output_qspec = qspec
        else:
            annotation_to_update.input_qspec_map[edge_or_node[0]] = qspec

    @staticmethod
    def _get_torch_ao_qspec_from_qp(qp: QuantizationPointBase) -> TorchAOQuantizationSpec:
        """
        Retrieves the quantization configuration from the given quantization point and
        converts it into a TorchAOQuantizationSpec.

        :param qp: An instance of QuantizationPointBase.
        :return: A TorchAOQuantizationSpec retrieved and converted from the quantization point.
        """
        # Eps value is copied from nncf/torch/quantization/layers.py
        extra_args = {"eps": 1e-16}
        qconfig = qp.qconfig
        is_weight = qp.is_weight_quantization_point()

        if qconfig.per_channel:
            torch_qscheme = (
                torch.per_channel_symmetric
                if qconfig.mode is QuantizationScheme.SYMMETRIC
                else torch.per_channel_affine
            )
        else:
            torch_qscheme = (
                torch.per_tensor_symmetric if qconfig.mode is QuantizationScheme.SYMMETRIC else torch.per_tensor_affine
            )
        if is_weight:
            observer = PerChannelMinMaxObserver
            quant_min = -128
            quant_max = 127
            dtype = torch.int8
            channel_axis = 0
        else:
            observer = (
                HistogramObserver
                if torch_qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]
                else PerChannelMinMaxObserver
            )
            quant_min = 0
            quant_max = 255
            dtype = torch.int8 if qconfig.signedness_to_force else torch.uint8
            channel_axis = 1  # channel dim for activations
        return TorchAOQuantizationSpec(
            dtype=dtype,
            observer_or_fake_quant_ctr=observer.with_args(**extra_args),
            quant_min=quant_min,
            quant_max=quant_max,
            qscheme=torch_qscheme,
            ch_axis=channel_axis,
            is_dynamic=False,
        )

    def validate(self, model: torch.fx.GraphModule) -> None:
        """
        Validates the annotated model before the insertion of FakeQuantizers / observers.

        :param model: Annotated torch.fx.GraphModule to validate after the  annotation.
        """
        pass

    def transform_for_annotation(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """
        Allows for user defined transforms to run before annotating the graph.
        This allows quantizer to allow quantizing part of the model that are otherwise not quantizable.
        For example quantizer can
        a) decompose a compound operator like scaled dot product attention,
        into bmm and softmax if quantizer knows how to quantize bmm/softmax but not sdpa
        or b) transform scalars to tensor to allow quantizing scalars.

        Note: this is an optional method

        :param model: Given torch.fx.GraphModule to transform before the annotation.
        :return: The transformed torch.fx.GraphModule ready for the annotation.
        """
        return model