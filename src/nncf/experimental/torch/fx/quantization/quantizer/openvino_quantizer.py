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
from enum import Enum
from typing import (
    Any,
    Optional,
)

import torch.fx
from observers import INT4WeightObserver
from observers import INT8WeightObserver
from torch.ao.quantization.observer import HistogramObserver
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from torch.ao.quantization.observer import UniformQuantizationObserverBase
from torch.ao.quantization.quantizer.quantizer import EdgeOrNode
from torch.ao.quantization.quantizer.quantizer import Quantizer as TorchAOQuantizer
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation as TorchAOQuantizationAnnotation,
)
from torchao.quantization.pt2e.quantizer import (
    QuantizationSpec as TorchAOQuantizationSpec,
)
from torchao.quantization.pt2e.quantizer import (
    QuantizationSpecBase as TorchAOQuantizationSpecBase,
)

import nncf
import nncf.common.quantization as quantization
import nncf.experimental.torch.fx as nncf_fx
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.utils.api_marker import api
from nncf.quantization.algorithms.weight_compression.config import (
    WeightCompressionParameters,
)
from nncf.quantization.quantize_model import get_weight_compression_configuration

QUANT_ANNOTATION_KEY = "quantization_annotation"


class QuantizationMode(Enum):
    """
    Defines special quantization modes.

    - INT8_SYM: INT8 symmetric quantization for both activations and weights.
    - INT8_MIXED: INT8 asymmetric quantization for activations, symmetric for weights.
    - INT8_TRANSFORMER: Optimized INT8 quantization for transformer-based models
    - INT8WO_SYM: INT8 symmetric quantization for weights only.
    - INT8WO_ASYM: INT8 asymmetric quantization for weights only.
    - INT4WO_SYM: INT4 symmetric quantization for weights only.
    - INT4WO_ASYM: INT4 asymmetric quantization for weights only
    """

    INT8_SYM = "int8_sym"
    INT8_MIXED = "int8_mixed"
    INT8_TRANSFORMER = "int8_transformer"
    INT8WO_SYM = "int8wo_sym"
    INT8WO_ASYM = "int8wo_asym"
    INT4WO_SYM = "int4wo_sym"
    INT4WO_ASYM = "int4wo_asym"


@api(canonical_alias="nncf.experimental.torch.fx.OpenVINOQuantizer")
class OpenVINOQuantizer(TorchAOQuantizer):
    """
    Implementation of the Torch AO quantizer which annotates models with quantization annotations
    optimally for the inference via OpenVINO.
    """

    WEIGHTS_ONLY_COMPRESSION_MODES = (
        QuantizationMode.INT4WO_SYM,
        QuantizationMode.INT4WO_ASYM,
        QuantizationMode.INT8WO_SYM,
        QuantizationMode.INT8WO_ASYM,
    )

    def __init__(
        self,
        *,
        mode: QuantizationMode = QuantizationMode.INT8_SYM,
        **kwargs,
    ):
        """
        :param mode: Defines special quantization modes.
            - INT8_SYM: INT8 symmetric quantization for both activations and weights.
            - INT8_MIXED: INT8 asymmetric quantization for activations, symmetric for weights.
            - INT8_TRANSFORMER: Optimized INT8 quantization for transformer-based models
            Default value is INT8_SYM.
            - INT4_SYM: Symmetric INT4 Weights-Only Compression
            - INT4_ASYM: Asymmetric INT4 Weights-Only Compression
        :param kwargs: Arguments to pass to the NNCF MinMaxQuantization algorithm.
        """
        self.mode = mode
        if self.mode not in OpenVINOQuantizer.WEIGHTS_ONLY_COMPRESSION_MODES:
            if mode == QuantizationMode.INT8_SYM:
                preset = nncf.quantization.structs.QuantizationPreset.PERFORMANCE
                model_type = None
            elif mode == QuantizationMode.INT8_MIXED:
                preset = quantization.structs.QuantizationPreset.MIXED
                model_type = None
            else:
                preset = None
                model_type = nncf.parameters.ModelType.TRANSFORMER
            self._algo = (
                nncf.quantization.algorithms.min_max.algorithm.MinMaxQuantization(
                    preset=preset, model_type=model_type, **kwargs
                )
            )
        else:
            self.weight_compression_configuration = (
                get_weight_compression_configuration(
                    mode.value.replace(
                        "wo", ""
                    ),  # Mode value has to match NNCF CompressWeightsMode
                    **kwargs,
                )
            )
            _weight_compression_configuration = self.weight_compression_configuration
            subset_size = 1  # Doesn't really matter in this case since it is data-free. Should just be +ve
            self._algo = nncf.quantization.algorithms.weight_compression.algorithm.WeightCompression(
                subset_size=subset_size, **_weight_compression_configuration
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
        self._algo.set_ignored_scope(
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

    def get_nodes_to_compress(self, model, nncf_graph) -> list[NNCFNode]:
        self._algo.set_backend_entity(model)
        return self._algo.get_nodes_to_compress(nncf_graph)

    def get_nncf_weight_compression_setup(
        self, model: torch.fx.GraphModule, nncf_graph: NNCFGraph
    ) -> quantization.quantizer_setup.SingleConfigQuantizerSetup:
        nodes_to_compress = self.get_nodes_to_compress(model, nncf_graph)
        return self._algo.get_weight_compression_parameters(
            model, nncf_graph, nodes_to_compress
        )[0]

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        nncf_graph = nncf_fx.nncf_graph_builder.GraphConverter.create_nncf_graph(model)
        graph = model.graph
        node_vs_torch_annotation: defaultdict[
            torch.fx.Node, TorchAOQuantizationAnnotation
        ] = defaultdict(TorchAOQuantizationAnnotation)

        if self.mode in OpenVINOQuantizer.WEIGHTS_ONLY_COMPRESSION_MODES:
            node_vs_torch_annotation = self._annotate_weight_compression(
                model, graph, nncf_graph, node_vs_torch_annotation
            )
        else:
            node_vs_torch_annotation = self._annotate_post_training_quantization(
                model, graph, nncf_graph, node_vs_torch_annotation
            )

        for node, annotation in node_vs_torch_annotation.items():
            assert QUANT_ANNOTATION_KEY not in node.meta
            node.meta[QUANT_ANNOTATION_KEY] = annotation

        return model

    @staticmethod
    def _get_unified_scales_root_quantizer_id(
        nncf_graph: NNCFGraph,
        quantizer_ids: list[int],
        quantizer_setup: quantization.quantizer_setup.SingleConfigQuantizerSetup,
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
            target_node_name = quantizer_setup.quantization_points[
                quantizer_id
            ].insertion_point.target_node_name
            nncf_node = nncf_graph.get_node_by_name(target_node_name)
            if (
                nncf_node_quantizer_id is None
                or nncf_node.node_id < nncf_node_quantizer_id
            ):
                root_quantizer_id = quantizer_id
                nncf_node_quantizer_id = nncf_node.node_id
        if root_quantizer_id is None:
            msg = "Root quantizer ids can't be None"
            raise nncf.InternalError(msg)
        return root_quantizer_id

    @staticmethod
    def _get_edge_or_node_and_annotation(
        graph: torch.fx.Graph,
        nncf_graph: NNCFGraph,
        qp: quantization.quantizer_setup.QuantizationPointBase,
        node_vs_torch_annotation: dict[torch.fx.Node, TorchAOQuantizationAnnotation],
    ) -> tuple[EdgeOrNode, TorchAOQuantizationAnnotation]:
        """
        Retrieves the edge or node and its corresponding TorchAOQuantizationAnnotation based on the given graph,
        quantization point, and node-to-annotation mapping.

        :param graph: torch.fx.Graph instance.
        :param nncf_graph: NNCFGraph instance.
        :param qp: QuantizationPointBase instance.
        :param node_vs_torch_annotation: A dictionary mapping torch.fx.GraphNode objects to their respective
            QuantizationAnnotations.
        :return: A tuple containing the EdgeOrNode and its associated TorchAOQuantizationAnnotation.
        """
        target_node = nncf_fx.node_utils.get_graph_node_by_name(
            graph, qp.insertion_point.target_node_name
        )
        annotation = node_vs_torch_annotation[target_node]
        edge_or_node = OpenVINOQuantizer._get_edge_or_node(target_node, qp, nncf_graph)
        return edge_or_node, annotation

    @staticmethod
    def _get_weight_edge(
        target_node: torch.fx.Node,
        nncf_graph: NNCFGraph,
    ) -> tuple[torch.fx.Node, torch.fx.Node]:
        """
        Returns the FX node corresponding to the weight tensor input of a given operator node.
        Uses the NNCF graph to identify which input port of the target node holds the weight.
        If multiple weight ports are present, a warning is issued and only the first one is used.

        :param target_node: FX node representing a weighted operation (e.g., Linear, Conv).
        :param nncf_graph: NNCFGraph used to determine weight port indices.
        :return: Edge represented by a Tuple of (weight_node, target_node), where weight_node is the FX node supplying
        the weight.
        """
        nncf_node = nncf_graph.get_node_by_name(target_node.name)
        weights_ports_ids = nncf.torch.model_graph_manager.get_weight_tensor_port_ids(
            nncf_node, nncf_graph
        )
        if len(weights_ports_ids) > 1:
            # TODO(dlyakhov): support quantization for nodes with several weights
            nncf.common.logging.nncf_logger.warning(
                f"Quantization of the weighted node {target_node.name}"
                " is not yet supported by the OpenVINOQuantizer."
                f" Only the weight on port ID {weights_ports_ids[0]} will be quantized."
                f" Quantizable weights are located on ports: {weights_ports_ids}."
            )
        weight_node = target_node.all_input_nodes[weights_ports_ids[0]]
        return (weight_node, target_node)

    @staticmethod
    def _get_edge_or_node(
        target_node: torch.fx.Node,
        qp: quantization.quantizer_setup.QuantizationPointBase,
        nncf_graph: NNCFGraph,
    ) -> EdgeOrNode:
        """
        Returns the edge or node based on the given target node and quantization point.

        :param target_node: Target node instance.
        :param qp: QuantizationPointBase instance.
        :param graph: NNCFGraph instance.
        :return: The corresponding EdgeOrNode derived from the target node and quantization point.
        """
        ip = qp.insertion_point
        if qp.is_weight_quantization_point():
            OpenVINOQuantizer._get_weight_edge(target_node, nncf_graph)

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
    def _get_torch_ao_qspec_from_nncf_config_for_wc(
        wc_param: WeightCompressionParameters,
    ) -> TorchAOQuantizationSpec:
        """
        Returns a TorchAO TorchAOQuantizationSpec based on NNCF weight compression parameter.

        :param wc_param: NNCF Weight compression parameters for the node.
        :return: A TorchAO TorchAOQuantizationSpec.
        """
        observer: type[UniformQuantizationObserverBase]

        extra_args: dict[str, Any] = {}

        qmode = wc_param.compression_config.mode
        extra_args["wc_param"] = wc_param
        is_asym_mode = wc_param.compression_config.is_asym_mode
        if qmode in [
            nncf.CompressWeightsMode.INT4_ASYM,
            nncf.CompressWeightsMode.INT4_SYM,
        ]:
            observer = INT4WeightObserver  # type: ignore[type-abstract]
            quant_min = -8 if not is_asym_mode else 0
            quant_max = 7 if not is_asym_mode else 15
            dtype = torch.int8
            channel_axis = 0
            torch_qscheme = torch_qscheme = (
                torch.per_channel_symmetric
                if not is_asym_mode
                else torch.per_channel_affine
            )
        else:
            observer = INT8WeightObserver  # type: ignore[type-abstract]
            quant_min = -128 if not is_asym_mode else 0
            quant_max = 127 if not is_asym_mode else 255
            dtype = torch.int8
            channel_axis = 0
            torch_qscheme = (
                torch.per_channel_symmetric
                if not is_asym_mode
                else torch.per_channel_affine
            )
        return TorchAOQuantizationSpec(
            dtype=dtype,
            observer_or_fake_quant_ctr=observer.with_args(**extra_args),
            quant_min=quant_min,
            quant_max=quant_max,
            qscheme=torch_qscheme,
            ch_axis=channel_axis,
            is_dynamic=False,
        )

    @staticmethod
    def _get_torch_ao_qspec_from_nncf_config_for_ptq(
        qp: quantization.quantizer_setup.QuantizationPointBase,
    ) -> TorchAOQuantizationSpec:
        """
        Returns a TorchAO TorchAOQuantizationSpec based on NNCF quantization point.

        :param qp: Quantization point from NNCF.
        :return: A TorchAO TorchAOQuantizationSpec.
        """
        observer: type[UniformQuantizationObserverBase]

        # Eps value is copied from nncf/torch/quantization/layers.py
        extra_args: dict[str, Any] = {"eps": 1e-16}

        is_weight = qp.is_weight_quantization_point()
        qconfig = qp.qconfig
        dtype = torch.int8
        quant_min = None
        quant_max = None
        channel_axis = None

        if qconfig.per_channel:
            torch_qscheme = (
                torch.per_channel_symmetric
                if qconfig.mode is quantization.structs.QuantizationScheme.SYMMETRIC
                else torch.per_channel_affine
            )
        else:
            torch_qscheme = (
                torch.per_tensor_symmetric
                if qconfig.mode is quantization.structs.QuantizationScheme.SYMMETRIC
                else torch.per_tensor_affine
            )
        if is_weight:
            observer = PerChannelMinMaxObserver
            quant_min = -128
            quant_max = 127
            dtype = torch.int8
            channel_axis = 0
            torch_qscheme = (
                torch.per_channel_symmetric
                if qconfig.mode is quantization.structs.QuantizationScheme.SYMMETRIC
                else torch.per_channel_affine
            )
        else:
            observer = (
                HistogramObserver
                if torch_qscheme
                in [torch.per_tensor_symmetric, torch.per_tensor_affine]
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

    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
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
