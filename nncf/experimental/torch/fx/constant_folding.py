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

import collections
from typing import Any, Callable, Dict, List, Optional

import torch.fx
import torch.utils._pytree as pytree

aten = torch.ops.aten


def _replace_node_with_constant(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    value: torch.Tensor,
) -> None:
    """
    Replaces given node to a constant with given value.

    :param gm: GraphModule instance.
    :param node: A node to replace with a constant.
    :param value: Value to use as a constant instead of the given node.
    """
    g = gm.graph

    if not hasattr(gm, "_frozen_param_count"):
        gm._frozen_param_count = 0  # type: ignore[assignment]
    i = gm._frozen_param_count

    while True:
        qualname = f"_frozen_param{i}"
        if not hasattr(gm, qualname):
            break
        i += 1

    gm._frozen_param_count = i + 1

    with g.inserting_before(node):
        new_input_node = g.create_node("get_attr", qualname, (), {})
        node.replace_all_uses_with(new_input_node)
        new_input_node.meta.update(node.meta)
        g.erase_node(node)

    # needed to suppress `does not reference an nn.Module, nn.Parameter, or buffer` warning
    gm.register_buffer(qualname, value)
    setattr(gm, qualname, value)


def _is_const_source(node: torch.fx.Node) -> bool:
    """
    Return True if node is a constant node.

    :param node: Given node.
    :return: True if node is a constant node.
    """
    return node.op == "get_attr"


class ConstantFolder(torch.fx.Interpreter):
    """
    Finds constant subraphs and place it in node_replacement attribute.
    """

    def __init__(
        self,
        gm: torch.fx.GraphModule,
    ) -> None:
        super().__init__(gm)
        self.node_replacements: Dict[torch.fx.Node, Any] = {}
        self.replaced_uses: Dict[torch.fx.Node, int] = collections.Counter()
        self.unknown_value = object()

        # overwrite this to deallocate env values if their only remaining use
        # is the output
        self.user_to_last_uses = self.node_to_last_non_output_use()

    def _deduce_value(self, node: torch.fx.Node) -> Any:
        return super().run_node(node)

    def is_impure(self, node: torch.fx.node.Node) -> bool:
        def is_woq_int8_pattern(node: torch.fx.node.Node) -> bool:
            return (
                node.target == torch.ops.prims.convert_element_type.default  # type: ignore[return-value]
                and isinstance(node.args[0], torch.fx.Node)
                and "val" in node.args[0].meta
                and node.args[0].meta["val"].dtype == torch.int8  # type: ignore[union-attr]
                and node.args[1] == torch.bfloat16
            )

        if (
            is_woq_int8_pattern(node)
            or (
                node.target == torch.ops.aten.permute.default
                and len(node.users) == 1
                and is_woq_int8_pattern(next(iter(node.users)))
            )
        ) and _is_const_source(node.args[0]):
            # Case 1: int8_weight -> dq -> bf16_weight
            # Case 2: int8_weight -> permute -> dq -> bf16_weight
            return True

        quant_registered = getattr(torch.ops.quantized_decomposed, "dequantize_per_channel", None) is not None
        if quant_registered and node.target in [
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
        ]:
            # For the pattern fp32_weight -> q -> dq
            # We only folding fp32_weight -> q
            # int8_weight and leave dq in graph to be fused
            return True
        return False

    def node_to_last_non_output_use(self) -> Dict[torch.fx.Node, List[torch.fx.Node]]:
        last_non_output_use = collections.defaultdict(list)
        seen_uses = set()
        output_node = next(iter(reversed(self.module.graph.nodes)))

        for node in reversed(self.module.graph.nodes):
            if node.target == "output":
                continue

            def add_use(inp: torch.fx.Node) -> None:
                if inp in seen_uses:
                    return

                seen_uses.add(inp)
                last_non_output_use[node].append(inp)

            # In-place is fine since we don't mutate
            pytree.tree_map_only_(torch.fx.Node, add_use, (node.args, node.kwargs))

            # if this node is only used in output, we want to gc it right away
            if len(node.users) == 1 and output_node in node.users:
                last_non_output_use[node].append(node)

        return last_non_output_use

    def run_node(self, node: torch.fx.Node) -> Any:
        if node.target == "output":
            # because we remove nodes from env on last non output use,
            # re-define them now or we'll get error in interpreter
            def set_env(arg: torch.fx.Node) -> None:
                self.env[arg] = self.unknown_value

            # In-place is fine since we don't mutate
            pytree.tree_map_only_(torch.fx.Node, set_env, node.args)
            return super().run_node(node)

        args, kwargs = self.fetch_args_kwargs_from_env(node)
        flattened_inputs = pytree.arg_tree_leaves(*args, **kwargs)

        # We need to do this weird thing because in cases where flattened_inputs
        # contains a ScriptObject, equality checking results in a type error if
        # the types are different.
        if any(
            type(self.unknown_value) is type(input_) and self.unknown_value == input_ for input_ in flattened_inputs
        ):
            return self.unknown_value

        if node.op == "call_function" and node.target == aten._efficientzerotensor.default:
            return self.unknown_value

        if node.op == "call_function" and node.name == "triton_kernel_wrapper_functional_proxy":
            return self.unknown_value

        # skip constructors, since inductor generates optimal code for them already
        # and turning into tensor would result in an additional global memory read
        if not _is_const_source(node) and not any(isinstance(e, torch.Tensor) for e in flattened_inputs):
            return self.unknown_value

        # All mutations should either be removed or on inputs which we did not make constant
        if isinstance(node.target, torch._ops.OpOverload) and torch.Tag.nondeterministic_seeded in node.target.tags:
            return self.unknown_value

        out = self._deduce_value(node)
        if out == self.unknown_value:
            return self.unknown_value

        if not _is_const_source(node) and isinstance(out, torch.Tensor):
            if out.device.type == "meta":
                return out

            if self.is_impure(node):
                return self.unknown_value

            self.add_node_replacement(node, out)

            flattened_node_inps = pytree.arg_tree_leaves(*node.args, **node.kwargs)

            for n in flattened_node_inps:
                if not isinstance(n, torch.fx.Node):
                    continue

                self.replaced_uses[n] += 1

            for to_delete in self.user_to_last_uses.get(node, []):
                if self.replaced_uses[to_delete] == len(to_delete.users):
                    self.node_replacements.pop(to_delete, None)

        return out

    def add_node_replacement(self, node: torch.fx.Node, tensor: torch.Tensor) -> None:
        self.node_replacements[node] = tensor

    def run(self) -> Any:  # type: ignore[override]
        env: Dict[torch.fx.Node, Any] = {}
        self.insert_placerholder_values(env)
        return super().run(initial_env=env)

    def insert_placerholder_values(self, env: Dict[torch.fx.Node, Any]) -> None:
        for n in self.module.graph.find_nodes(op="placeholder"):
            env[n] = self.unknown_value  # type: ignore[assignment]


def _is_impure(node: torch.fx.Node) -> bool:
    """
    Returns True if the node call affects the model outputs even in case
    the node have zero users, False otherwise.

    :param node: A node to check.
    :return: True if the node call affects the model outputs even in case
        the node have zero users, False otherwise.

    """
    return node.op in {"placeholder", "output"}


def constant_fold(
    gm: torch.fx.GraphModule,
    constraint_fn: Optional[Callable[[torch.fx.Node], bool]] = None,
) -> None:
    """
    Calcualtes constant subgraphs values and replaces them with a constant node inplace.

    :param gm: Given graph model.
    :param constraint_fn: Constraint function which takes a node and returs the constraint:
        should the node be constant folded or not.
    """
    with torch.no_grad():
        with torch.utils._python_dispatch._disable_current_modes():
            cf = ConstantFolder(gm)
            cf.run()

            for node, constant in cf.node_replacements.items():
                if constraint_fn is not None and not constraint_fn(node):
                    continue
                _replace_node_with_constant(gm, node, constant)

            erased_params = []
            for node in gm.graph.find_nodes(op="get_attr"):
                if len(node.users) == 0:
                    if hasattr(gm, node.target):
                        delattr(gm, node.target)
                    erased_params.append(node)

            for node in erased_params:
                gm.graph.erase_node(node)

            # Custom _is_impure function allows to eliminate all layers with zero
            # users including inplace ops like relu_ besides output and placeholders.
            gm.graph.eliminate_dead_code(_is_impure)
            gm.graph.lint()
            gm.recompile()
