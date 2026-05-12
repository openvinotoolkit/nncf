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
from typing import Any, TypeVar

from torch import nn

import nncf
import nncf.torch.graph.operator_metatypes as om
from nncf.common.logging import nncf_logger
from nncf.parameters import PruneMode
from nncf.scopes import IgnoredScope
from nncf.scopes import get_ignored_node_names_from_ignored_scope
from nncf.torch.function_hook.nncf_graph.layer_attributes import PT2OpLayerAttributes
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import build_nncf_graph
from nncf.torch.function_hook.pruning.magnitude.algo import apply_magnitude_pruning
from nncf.torch.function_hook.pruning.rb.algo import apply_regularization_based_pruning
from nncf.torch.function_hook.wrapper import wrap_model
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.model_graph_manager import get_const_node

TModel = TypeVar("TModel", bound=nn.Module)

OPERATORS_WITH_WEIGHTS_METATYPES = [
    om.PTConv1dMetatype,
    om.PTConv2dMetatype,
    om.PTConv3dMetatype,
    om.PTConvTranspose1dMetatype,
    om.PTConvTranspose2dMetatype,
    om.PTConvTranspose3dMetatype,
    om.PTDepthwiseConv1dSubtype,
    om.PTDepthwiseConv2dSubtype,
    om.PTDepthwiseConv3dSubtype,
    om.PTEmbeddingBagMetatype,
    om.PTEmbeddingMetatype,
    om.PTLinearMetatype,
    om.PTMatMulMetatype,
]


def prune(
    model: TModel,
    mode: PruneMode,
    ratio: float | None = None,
    ignored_scope: IgnoredScope | None = None,
    examples_inputs: Any | None = None,
) -> TModel:
    if examples_inputs is None:
        msg = "`sparsity` function requires `examples_inputs` argument to be specified for Torch backend"
        raise nncf.InternalError(msg)

    model = wrap_model(model)
    graph = build_nncf_graph(model, examples_inputs)
    parameters_to_sparsity = get_prunable_parameters(graph, ignored_scope)

    # Select and apply the pruning algorithm by mode
    if mode in [PruneMode.UNSTRUCTURED_MAGNITUDE_GLOBAL, PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL]:
        if ratio is None:
            msg = f"`ratio` parameter should be specified for {mode} mode in nncf.prune function"
            raise nncf.InternalError(msg)
        model = apply_magnitude_pruning(model, parameters_to_sparsity, mode, ratio)
    elif mode == PruneMode.UNSTRUCTURED_REGULARIZATION_BASED:
        if ratio is not None:
            nncf_logger.warning(
                f"`ratio` parameter is ignored for {mode} mode in nncf.prune function. "
                "Target pruning ratio should be set by RBLoss."
            )
        model = apply_regularization_based_pruning(model, parameters_to_sparsity)
    else:
        msg = f"Pruning mode {mode} is not implemented for Torch backend"
        raise nncf.InternalError(msg)
    return model


def get_prunable_parameters(graph: PTNNCFGraph, ignored_scope: IgnoredScope | None) -> set[str]:
    """
    Retrieves the names of constant nodes that are associated with prunable parameters in the given graph.

    :param graph: The NNCFGraph to analyze.
    :param ignored_scope: The IgnoredScope to filter out nodes.
    :return: A set of names of constant nodes that are associated with prunable parameters.
    """
    ignored_names: set[str] = set()
    if ignored_scope is not None:
        ignored_names = get_ignored_node_names_from_ignored_scope(ignored_scope, graph)

    nodes_with_weights = graph.get_nodes_by_metatypes(OPERATORS_WITH_WEIGHTS_METATYPES)
    parameters_to_sparsity: set[str] = set()
    for node in nodes_with_weights:
        if node.node_name in ignored_names:
            continue
        layer_attributes = node.layer_attributes
        if not isinstance(layer_attributes, PT2OpLayerAttributes):
            msg = f"Expected PT2OpLayerAttributes, got {type(layer_attributes)} for node {node.node_name}"
            raise nncf.InternalError(msg)
        metatype = node.metatype
        if not issubclass(metatype, om.PTOperatorMetatype):
            msg = f"Expected PTOperatorMetatype, got {type(metatype)} for node {node.node_name}"
            raise nncf.InternalError(msg)
        weights_ports = layer_attributes.constant_port_ids.intersection(metatype.weight_port_ids)

        for port in weights_ports:
            const_node = get_const_node(node, port, graph)
            if const_node is not None and const_node.node_name not in ignored_names:
                parameters_to_sparsity.add(const_node.node_name)

    return parameters_to_sparsity
