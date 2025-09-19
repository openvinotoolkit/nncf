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
from typing import Any, Optional

import torch
from torch import nn

import nncf
import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph.graph import NNCFNode
from nncf.common.utils.helpers import create_table
from nncf.parameters import PruneMode
from nncf.scopes import IgnoredScope
from nncf.scopes import get_ignored_node_names_from_ignored_scope
from nncf.torch.function_hook.hook_storage import decode_hook_name
from nncf.torch.function_hook.nncf_graph.layer_attributes import PT2OpLayerAttributes
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import build_nncf_graph
from nncf.torch.function_hook.prune.magnitude.modules import UnstructuredPruneBinaryMask
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.function_hook.wrapper import register_post_function_hook
from nncf.torch.function_hook.wrapper import wrap_model
from nncf.torch.model_graph_manager import get_const_data
from nncf.torch.model_graph_manager import get_const_data_by_name
from nncf.torch.model_graph_manager import get_const_node

OPERATORS_WITH_WEIGHTS_METATYPES = [
    om.PTConv1dMetatype,
    om.PTConv2dMetatype,
    om.PTConv3dMetatype,
    om.PTDepthwiseConv1dSubtype,
    om.PTDepthwiseConv2dSubtype,
    om.PTDepthwiseConv3dSubtype,
    om.PTLinearMetatype,
    om.PTConvTranspose1dMetatype,
    om.PTConvTranspose2dMetatype,
    om.PTConvTranspose3dMetatype,
    om.PTEmbeddingMetatype,
    om.PTEmbeddingBagMetatype,
]


def prune(
    model: nn.Module,
    mode: PruneMode,
    ratio: float,
    ignored_scope: Optional[IgnoredScope] = None,
    examples_inputs: Optional[Any] = None,
) -> nn.Module:
    if examples_inputs is None:
        msg = "`sparsity` function requires `examples_inputs` argument to be specified for Torch backend"
        raise nncf.InternalError(msg)

    model = wrap_model(model)
    graph = build_nncf_graph(model, examples_inputs)

    ignored_names: set[str] = set()
    if ignored_scope is not None:
        ignored_names = get_ignored_node_names_from_ignored_scope(ignored_scope, graph)
    nodes_with_weights = graph.get_nodes_by_metatypes(OPERATORS_WITH_WEIGHTS_METATYPES)

    # Get nodes by graph and ignored scope
    const_nodes_to_sparsity: list[NNCFNode] = []
    for node in nodes_with_weights:
        if node.node_name in ignored_names:
            continue
        layer_attributes = node.layer_attributes
        if not isinstance(layer_attributes, PT2OpLayerAttributes):
            msg = f"Expected PT2OpLayerAttributes, got {type(layer_attributes)} for node {node.node_name}"
            raise nncf.InternalError(msg)
        weights_ports = layer_attributes.constant_port_ids.intersection(node.metatype.weight_port_ids)

        const_nodes_to_sparsity.extend(get_const_node(node, port, graph) for port in weights_ports)

    # Insert hooks
    pruned_param_names = set()
    for node in const_nodes_to_sparsity:
        if node.node_name in pruned_param_names:
            # To avoid adding multiple hooks to a shared parameters
            continue
        pruned_param_names.add(node.node_name)
        param_data = get_const_data(node, model)
        hook_module = UnstructuredPruneBinaryMask(node.layer_attributes.shape).to(device=param_data.device)
        register_post_function_hook(
            model=model,
            op_name=node.node_name,
            port_id=0,
            hook=hook_module,
        )

    # Set ratio
    prune_update_ratio(model, mode=mode, ratio=ratio)

    return model


def prune_update_ratio(
    model: nn.Module,
    mode: PruneMode,
    ratio: float,
) -> None:
    hook_storage = get_hook_storage(model)
    sparsity_module_map: dict[str, UnstructuredPruneBinaryMask] = {}

    for name, hook in hook_storage.named_hooks():
        if not isinstance(hook, UnstructuredPruneBinaryMask):
            continue

        hook_type, op_name, port_id = decode_hook_name(name)

        if hook_type != "post_hooks" or port_id != 0:
            msg = f"Unexpected place of SparsityBinaryMask: {hook_type=}, {op_name=}, {port_id=}"
            raise nncf.InternalError(msg)

        sparsity_module_map[op_name] = hook

    if not sparsity_module_map:
        msg = "No found Sparsity modules in the model"
        return nncf.InternalError(msg)

    if mode == PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL:
        for const_name, hook in sparsity_module_map.items():
            data = get_const_data_by_name(const_name, model)

            # Calculate threshold and binary mask
            with torch.no_grad():
                threshold_index = int((data.numel() - 1) * ratio)
                abs_data = torch.abs(data)
                threshold = abs_data.view(-1).kthvalue(threshold_index).values
                new_mask = (abs_data >= threshold).to(dtype=torch.bool)

            # Set new mask
            hook.binary_mask = new_mask

    elif mode == PruneMode.UNSTRUCTURED_MAGNITUDE_GLOBAL:
        # Get threshold value for all normalized weights
        all_weights: list[torch.Tensor] = []
        for const_name, hook in sparsity_module_map.items():
            data = get_const_data_by_name(const_name, model)

            with torch.no_grad():
                all_weights.append((torch.abs(data) / data.norm(2)).view(-1))

        cat_all_weights = torch.cat(all_weights).view(-1)
        threshold_index = int((cat_all_weights.numel() - 1) * ratio)
        threshold_val = cat_all_weights.kthvalue(threshold_index).values

        for const_name, hook in sparsity_module_map.items():
            data = get_const_data_by_name(const_name, model)

            # Calculate threshold and binary mask
            with torch.no_grad():
                threshold_index = int((data.numel() - 1) * ratio)
                norm_data = torch.abs(data) / data.norm(2)
                threshold = norm_data.view(-1).kthvalue(threshold_index).values
                new_mask = (norm_data >= threshold_val).to(dtype=torch.bool)

            # Set new mask
            hook.binary_mask = new_mask


def prune_stat(model: nn.Module) -> None:
    hook_storage = get_hook_storage(model)

    all_parameters = sum(p.numel() for p in model.parameters())
    zeroes_count = 0
    pruned_tensors_size = 0
    per_tensor_stats: dict[str, tuple[Any]] = {}

    for name, hook in hook_storage.named_hooks():
        if not isinstance(hook, UnstructuredPruneBinaryMask):
            continue
        _, op_name, _ = decode_hook_name(name)

        mask_size = hook.binary_mask.numel()
        pruned_tensors_size += mask_size
        zeroes = mask_size - int(torch.count_nonzero(hook.binary_mask).item())
        zeroes_count += zeroes
        per_tensor_stats[op_name] = (
            op_name,
            tuple(hook.binary_mask.shape),
            zeroes / mask_size,
            mask_size / all_parameters,
        )

    model_string = create_table(
        header=["Statistic's name", "Value"],
        rows=[
            ["Pune ratio of the whole model", zeroes_count / (all_parameters or 1)],
            ["Pune ratio of all pruned layers", zeroes_count / (pruned_tensors_size or 1)],
        ],
    )

    layers_string = create_table(
        header=[
            "Layer's name",
            "Weight's shape",
            "Prune ratio",
            "Weight's percentage",
        ],
        rows=per_tensor_stats.values(),
    )

    pretty_string = f"Statistics by pruned layers:\n{layers_string}\n\nStatistics of the pruned model:\n{model_string}"
    print(pretty_string)
