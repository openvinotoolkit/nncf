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

from typing import TypeVar, cast

import torch
from torch import nn

import nncf
from nncf.common.graph.graph import NNCFNode
from nncf.common.logging import nncf_logger
from nncf.parameters import PruneMode
from nncf.torch.function_hook.hook_storage import decode_hook_name
from nncf.torch.function_hook.pruning.magnitude.modules import UnstructuredPruningMask
from nncf.torch.function_hook.pruning.magnitude.pattern import STRUCTURED_PRUNING_SUPPORTED_METATYPES
from nncf.torch.function_hook.pruning.magnitude.pattern import get_sparsity_pattern
from nncf.torch.function_hook.pruning.magnitude.pattern import get_structured_pattern_groups
from nncf.torch.function_hook.pruning.magnitude.structured_modules import StructuredPruningMask
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.function_hook.wrapper import register_post_function_hook
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import PTOperatorMetatype
from nncf.torch.model_graph_manager import get_const_data_by_name

TModel = TypeVar("TModel", bound=nn.Module)


def apply_magnitude_pruning(
    model: TModel,
    parameters: set[str],
    mode: PruneMode,
    ratio: float | None,
    graph: PTNNCFGraph | None = None,
) -> TModel:
    """
    Prunes the specified parameters of the given model using magnitude-based pruning.

    This function registers hooks to the model's parameters to apply a binary mask
    for magnitude pruning and update the specified ratio. Depending on the mode, either
    unstructured or structured (M:N sparsity pattern) masks are applied.

    :param model: The neural network model to be pruned.
    :param parameters: A set of parameter names to be pruned.
    :param mode: The mode of pruning to be applied.
    :param ratio: The ratio of parameters to prune. It is ignored for structured pruning
        modes with a fixed sparsity pattern.
    :param graph: The NNCFGraph of the model. It is used by structured pruning modes
        to determine the layer types of the pruned parameters.
    :returns: The pruned model with hooks registered for the specified parameters.
    """
    is_structured = mode == PruneMode.STRUCTURED_MAGNITUDE_2_4
    if is_structured:
        if graph is None:
            msg = f"`graph` parameter should be specified for {mode} mode in nncf.prune function"
            raise nncf.InternalError(msg)
        pattern = get_sparsity_pattern(mode)
        for param_name in parameters:
            param_data = get_const_data_by_name(param_name, model)
            # The parameter name refers to a constant node; the consuming operation node
            # determines the metatype that defines the weight layout of the tensor.
            consumer_node = get_structured_pruning_consumer_node(param_name, graph)
            if consumer_node is None or consumer_node.metatype not in STRUCTURED_PRUNING_SUPPORTED_METATYPES:
                consumer_node_name = consumer_node.node_name if consumer_node is not None else "not found"
                consumer_node_metatype = consumer_node.metatype if consumer_node is not None else None
                nncf_logger.warning(
                    f"Structured pruning with the {pattern} pattern is not supported for the parameter "
                    f"{param_name} of the operation {consumer_node_name} with metatype "
                    f"{consumer_node_metatype}. This parameter will be skipped."
                )
                continue
            register_post_function_hook(
                model=model,
                op_name=param_name,
                port_id=0,
                hook=StructuredPruningMask(tuple(param_data.shape)).to(device=param_data.device),
            )
    else:
        for param_name in parameters:
            param_data = get_const_data_by_name(param_name, model)
            register_post_function_hook(
                model=model,
                op_name=param_name,
                port_id=0,
                hook=UnstructuredPruningMask(tuple(param_data.shape)).to(device=param_data.device),
            )

    update_pruning_ratio(model, mode, ratio, graph)

    return model


def get_structured_pruning_consumer_node(param_name: str, graph: PTNNCFGraph) -> NNCFNode | None:
    """
    Returns the first operation node that consumes the constant parameter with the given name.

    :param param_name: The name of the constant node associated with the parameter.
    :param graph: The NNCFGraph of the model.
    :return: The first operation node consuming the constant, or None if the constant has no consumers.
    """
    const_node = graph.get_node_by_name(param_name)
    next_nodes = graph.get_next_nodes(const_node)
    return next_nodes[0] if next_nodes else None


def get_pruned_modules(model: nn.Module) -> dict[str, UnstructuredPruningMask | StructuredPruningMask]:
    """
    Retrieves a mapping of operation names to their corresponding
    magnitude pruning mask hooks from the given model.

    :param model: The model from which to retrieve the prunable modules.
    :return: A dictionary mapping tensor names to their corresponding UnstructuredPruningMask
        or StructuredPruningMask instances.
    """
    hook_storage = get_hook_storage(model)
    pruned_modules: dict[str, UnstructuredPruningMask | StructuredPruningMask] = dict()

    for name, hook in hook_storage.named_hooks():
        if not isinstance(hook, (UnstructuredPruningMask, StructuredPruningMask)):
            continue

        hook_type, op_name, port_id = decode_hook_name(name)
        if hook_type != "post_hooks" or port_id != 0:
            msg = f"Unexpected place of UnstructuredPruningMask: {hook_type=}, {op_name=}, {port_id=}"
            raise nncf.InternalError(msg)
        pruned_modules[op_name] = hook

    return pruned_modules


@torch.no_grad()
def update_pruning_ratio(
    model: nn.Module,
    mode: PruneMode,
    ratio: float | None,
    graph: PTNNCFGraph | None = None,
) -> None:
    """
    Updates masks with new pruning ratio for the given model based on the specified pruning mode.

    This function modifies the binary masks of the sparsity modules in the model according to the
    specified pruning strategy. It calculates the threshold for pruning based on the absolute values of the weights
    and updates the binary masks accordingly.

    For structured pruning modes with a fixed M:N sparsity pattern, `ratio` is ignored: the number of
    pruned weights in every group is determined by the pattern itself.

    :param model: The neural network model to be pruned.
    :param mode: The mode of pruning to be applied.
    :param ratio: The desired pruning ratio, represented as a float between 0 and 1.
    :param graph: The NNCFGraph of the model. It is used by structured pruning modes
        to determine the layer types of the pruned parameters.
    """
    pruned_modules_map = get_pruned_modules(model)

    if not pruned_modules_map:
        msg = "No pruned tensors found in the model"
        raise nncf.InternalError(msg)

    if mode == PruneMode.STRUCTURED_MAGNITUDE_2_4:
        if graph is None:
            msg = f"`graph` parameter should be specified for {mode} mode in nncf.prune function"
            raise nncf.InternalError(msg)
        pattern = get_sparsity_pattern(mode)
        for const_name, hook in pruned_modules_map.items():
            data = get_const_data_by_name(const_name, model)
            consumer_node = get_structured_pruning_consumer_node(const_name, graph)
            if consumer_node is None or consumer_node.metatype not in STRUCTURED_PRUNING_SUPPORTED_METATYPES:
                continue
            metatype = cast(type[PTOperatorMetatype], consumer_node.metatype)
            weight_port_id = metatype.weight_port_ids[0]
            groups = get_structured_pattern_groups(data, metatype, weight_port_id, pattern)
            # Keep the N-M weights with the largest magnitudes in every group of N weights.
            # The stable sort makes the selection deterministic for tied magnitudes:
            # the weights that come first along the structural dimension are retained.
            magnitudes = torch.abs(groups)
            sorted_indices = torch.argsort(magnitudes, dim=2, descending=True, stable=True)
            keep_mask = torch.zeros_like(magnitudes, dtype=torch.bool)
            keep_mask.scatter_(2, sorted_indices[:, :, : pattern.n - pattern.m], True)
            hook.binary_mask.copy_(keep_mask.reshape_as(data))

    elif mode == PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL:
        if ratio is None:
            msg = f"`ratio` parameter should be specified for {mode} mode in nncf.prune function"
            raise nncf.InternalError(msg)
        for const_name, hook in pruned_modules_map.items():
            data = get_const_data_by_name(const_name, model)

            # Calculate threshold and binary mask
            threshold_index = int((data.numel() - 1) * ratio)
            abs_data = torch.abs(data)
            threshold = abs_data.view(-1).kthvalue(threshold_index + 1).values
            new_mask = (abs_data > threshold).to(dtype=torch.bool)

            # Set new mask
            hook.binary_mask.copy_(new_mask)

    elif mode == PruneMode.UNSTRUCTURED_MAGNITUDE_GLOBAL:
        if ratio is None:
            msg = f"`ratio` parameter should be specified for {mode} mode in nncf.prune function"
            raise nncf.InternalError(msg)

        # Get threshold value for all normalized weights
        all_weights: list[torch.Tensor] = []
        for const_name in pruned_modules_map:
            data = get_const_data_by_name(const_name, model)
            all_weights.append((torch.abs(data) / data.norm(2)).view(-1))

        cat_all_weights = torch.cat(all_weights).view(-1)
        threshold_index = int((cat_all_weights.numel() - 1) * ratio)
        threshold_val = cat_all_weights.kthvalue(threshold_index + 1).values

        for const_name, hook in pruned_modules_map.items():
            data = get_const_data_by_name(const_name, model)

            # Calculate threshold and binary mask
            norm_data = torch.abs(data) / data.norm(2)
            new_mask = (norm_data > threshold_val).to(dtype=torch.bool)

            # Set new mask
            hook.binary_mask.copy_(new_mask)
    else:
        msg = f"Unsupported pruning mode: {mode}"
        raise nncf.InternalError(msg)
