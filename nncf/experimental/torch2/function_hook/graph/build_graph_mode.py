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

from __future__ import annotations

from typing import Any, Dict, Tuple, Union, cast

import networkx as nx  # type: ignore[import-untyped]
import torch
from torch import nn

from nncf.common.logging import nncf_logger as logger
from nncf.experimental.torch2.function_hook.graph.graph_utils import ConstMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import EdgeMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import FunctionMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import InOutMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import NodeType
from nncf.experimental.torch2.function_hook.graph.graph_utils import TensorInfo
from nncf.experimental.torch2.function_hook.graph.graph_utils import TensorMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import TensorSource
from nncf.experimental.torch2.function_hook.hook_executor_mode import FunctionHookMode
from nncf.experimental.torch2.function_hook.hook_executor_mode import OpMeta
from nncf.experimental.torch2.function_hook.hook_storage import HookStorage
from nncf.experimental.torch2.function_hook.weak_map import WeakUnhashableKeyMap
from nncf.experimental.torch2.function_hook.wrapper import ForwardWithHooks
from nncf.experimental.torch2.function_hook.wrapper import get_hook_storage


class GraphBuilderMode(FunctionHookMode):
    """
    A class that extends the `FunctionHookMode` and constructs nx.MultiDiGraph representing
    the operations within a PyTorch model. Each operation in the model is represented as a node in the graph.

    :param next_node_id: The index of the last added node in the graph.
    :param graph: An instance of `networkx.MultiDiGraph` used to represent the graph of model.
    :param tensor_info: A dictionary mapping tensor objects to tensor info.
    """

    def __init__(self, model: nn.Module, hook_storage: HookStorage):
        """
        Initialize the GraphBuilderMode.

        :param model: The PyTorch model to which the hooks will be applied.
        :param hook_storage: Storage for hooks to be executed.
        """
        super().__init__(model=model, hook_storage=hook_storage)
        self.next_node_id: int = 0
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.tensor_info: WeakUnhashableKeyMap[Union[torch.Tensor, torch.nn.Parameter], TensorInfo] = (
            WeakUnhashableKeyMap()
        )

        for name, parameter in self.model.named_parameters():
            self.tensor_info[parameter] = TensorInfo(
                tensor_source=TensorSource.parameter,
                shape=tuple(parameter.shape),
                dtype=parameter.dtype,
                output_port_id=0,
                source_node_id=None,
                name_in_model=name,
            )
        for name, buffer in self.model.named_buffers():
            self.tensor_info[buffer] = TensorInfo(
                tensor_source=TensorSource.buffer,
                shape=tuple(buffer.shape),
                dtype=buffer.dtype,
                output_port_id=0,
                source_node_id=None,
                name_in_model=name,
            )

    def __enter__(self) -> GraphBuilderMode:
        """
        Overload __enter__ to get correct return type hint.
        """
        super().__enter__()
        return self

    def register_new_node_id(self) -> int:
        """
        Generates and returns a unique id for a new node in the graph.

        :return: A unique id for a new node.
        """
        key = self.next_node_id
        self.next_node_id += 1
        return key

    def register_node_for_model_input_tensor(self, node_name: str, tensor: torch.Tensor) -> None:
        """
        Register a node for model input.

        :param node_name: The name of the input node to be registered.
        :param tensor: The tensor associated with the model's input.
        """
        node_id = self.register_new_node_id()
        with self.disable():
            self.tensor_info[tensor] = TensorInfo(
                tensor_source=TensorSource.input,
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
                output_port_id=0,
                source_node_id=node_id,
                name_in_model=None,
            )
            self.graph.add_node(node_id, type=NodeType.input, meta=InOutMeta.from_tensor(tensor, node_name))
        logger.debug(f"GraphBuilderMode.register_node_for_model_input_tensor: {node_id=} {node_name=}")

    def execute_hooks_for_model_input(self, name: str, value: Any) -> Any:
        """
        Overloaded function to register a model input to the graph.

        :param name: The name of the input argument.
        :param value: The value of the input argument.
        :return: The processed value after the hook is executed.
        """
        if isinstance(value, torch.Tensor):
            self.register_node_for_model_input_tensor(name, value)
        return super().execute_hooks_for_model_input(name, value)

    def register_node_for_model_output_tensor(self, node_name: str, tensor: torch.Tensor) -> None:
        """
        Registers a node for a model's output tensor in graph, creating an edge between
        the tensor's source node and the output node.

        :param node_name: The name of the output node to be registered.
        :param tensor: The tensor associated with the model's output.
        """
        node_id = self.register_new_node_id()
        tensor_info = self.tensor_info.get(tensor)
        if tensor_info is not None and isinstance(tensor_info.output_port_id, int):
            with self.disable():
                self.graph.add_edge(
                    tensor_info.source_node_id,
                    node_id,
                    meta=EdgeMeta.from_tensor(tensor, input_port=0, output_port=tensor_info.output_port_id),
                )
                self.graph.add_node(node_id, type=NodeType.output, meta=InOutMeta.from_tensor(tensor, node_name))
        logger.debug(f"GraphBuilderMode.register_node_for_model_input_tensor: {node_id=} {node_name=}")

    def execute_hooks_for_model_output(self, name: str, value: Any) -> Any:
        """
        Overloaded function to register a model output to the graph.

        :param name: The name of the input argument.
        :param value: The value of the input argument.
        :return: The processed value after the hook is executed.
        """
        value = super().execute_hooks_for_model_output(name, value)
        if isinstance(value, torch.Tensor):
            self.register_node_for_model_output_tensor(name, value)
        return value

    def execute_pre_hooks(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any], op_meta: OpMeta
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Overloaded function to register a model output to the graph.

        :param args: The arguments to the function.
        :param kwargs: The keyword arguments to the function.
        :param op_meta: Metadata for the operation.
        :return: The modified arguments and keyword arguments after pre-hooks.
        """
        _args, _kwargs = super().execute_pre_hooks(args, kwargs, op_meta)
        self.register_op_node(_args, _kwargs, op_meta)
        return _args, _kwargs

    def process_tensor_attributes(self, output: torch.Tensor, op_meta: OpMeta) -> None:
        """
        Processes special functions like attribute access on tensors using grad_fn.

        Process attributes: detect .T and .mT attributes of tensor
            .T - permute((1, 0))
            .mT - transpose(-2, -1)

        :param output: The output tensor.
        :param op_meta: Metadata about the operation.
        """
        fn_name = None
        fn_kwargs = None

        if output.grad_fn is not None:
            if output.grad_fn.name() == "TransposeBackward0":
                fn_name = "transpose"
                # grad_fn collect arguments as _saved_dim0=18446744073709551614
                fn_kwargs = {
                    "dim0": -(2**64 - output.grad_fn._saved_dim0),  # type: ignore[attr-defined]
                    "dim1": -(2**64 - output.grad_fn._saved_dim1),  # type: ignore[attr-defined]
                }
            if output.grad_fn.name() == "PermuteBackward0":
                fn_name = "permute"
                fn_kwargs = {"dims": output.grad_fn._saved_dims}  # type: ignore[attr-defined]

        if fn_name is not None and fn_kwargs is not None:
            self.graph.nodes[op_meta.extra_info["node_id"]]["meta"].fn_name = fn_name
            self.graph.nodes[op_meta.extra_info["node_id"]]["meta"].kwargs = fn_kwargs

    def execute_post_hooks(self, outputs: Any, op_meta: OpMeta) -> Any:
        """
        Overload execute_post_hooks to correct registered node for operation.
        Process __get__ function, to detect permute and transpose operation
        and remove node if operation return not tensor.
        """
        if op_meta.func.__name__ == "__get__":
            if isinstance(outputs, torch.Tensor):
                self.process_tensor_attributes(outputs, op_meta)
            else:
                # Remove the node corresponding to this operation from the graph, as non-tensor
                # outputs (like `tensor.shape` or similar) are not relevant for further algorithmic use.
                self.graph.remove_node(op_meta.extra_info["node_id"])
        outputs = super().execute_post_hooks(outputs, op_meta)
        return outputs

    def execute_hooks_for_parameter(self, value: torch.Tensor) -> torch.Tensor:
        """
        Overload execute_hooks_for_parameter to register parameters to the graph.

        :param value: The tensor to which the post-hook will be applied.
        :return: The processed tensor with the applied post-hook, if applicable.
        """
        tensor_info = self.tensor_info.get(value)
        if (
            tensor_info is not None
            and tensor_info.source_node_id is None
            and tensor_info.tensor_source in [TensorSource.buffer, TensorSource.parameter]
            and isinstance(tensor_info.name_in_model, str)
        ):
            node_id = self.register_new_node_id()
            with self.disable():
                self.graph.add_node(
                    node_id,
                    type=NodeType.const,
                    meta=ConstMeta.from_tensor(tensor=value, name_in_model=tensor_info.name_in_model),
                )
            tensor_info.source_node_id = node_id
            logger.debug(f"GraphBuilderMode._maybe_add_node_for_parameters: {node_id=} {tensor_info.name_in_model=}")
        return super().execute_hooks_for_parameter(value)

    def register_op_input_tensor(self, tensor: torch.Tensor, op_node_id: int, port_id: int, op_meta: OpMeta) -> None:
        """
        Registers a tensor input by creating an edge between the tensor's source node and the current operation node.

        :param tensor: Input tensor for the operation.
        :param node_id: Id if operation node.
        :param port_id: Port id of input argument.
        :param op_meta: Metadata about the operation.
        """
        tensor_info = self.tensor_info.get(tensor, None)
        if tensor_info is None:
            logger.debug(f"GraphBuilderMode.register_op_input_tensor: unknown tensor {op_meta.op_name=} {port_id=}")
            return
        self.graph.add_edge(
            tensor_info.source_node_id,
            op_node_id,
            meta=EdgeMeta.from_tensor(tensor, input_port=port_id, output_port=tensor_info.output_port_id),
        )

    def register_op_input(self, arg: Any, node_id: int, port_id: int, op_meta: OpMeta) -> Any:
        """
        Registers an operation input. Handles tensors or collections of tensors on one input port.

        :param arg: Operation input argument.
        :param node_id: Id if operation node.
        :param port_id: Port id of input argument.
        :param op_meta: Metadata about the operation.
        :return: Descriptor of the input. For a Tensor, this is a `TensorMeta` object.
             For a collection of Tensors, a collection of `TensorMeta` objects is returned.
             For other types, the original input `arg` is returned as-is.
        """
        if isinstance(arg, torch.Tensor):
            self.register_op_input_tensor(arg, node_id, port_id, op_meta)
            return TensorMeta.from_tensor(arg)
        elif isinstance(arg, (list, tuple, set)):
            op_attr = []
            for x in arg:
                if isinstance(x, torch.Tensor):
                    self.register_op_input_tensor(x, node_id, port_id, op_meta)
                    op_attr.append(TensorMeta.from_tensor(x))
                else:
                    op_attr.append(x)
            return op_attr
        return arg

    def register_op_node(self, args: Tuple[Any], kwargs: Dict[str, Any], op_meta: OpMeta) -> None:
        """
        Registers a new operation node in the graph.

        This method is responsible for:
            - adding a new node in the graph for the given operation and storing metadata such as the node ID,
              operation name, input arguments and keyword arguments;
            - adding edge associated with registered tensor.

        :param args: Positional arguments for the operation.
        :param kwargs: Keyword arguments for the operation.
        :param op_meta: Metadata about the operation, including the function being invoked.
        """
        node_id = self.register_new_node_id()
        op_meta.extra_info["node_id"] = node_id
        op_name = op_meta.op_name

        op_attrs = []
        op_kwargs = {}
        for port_id, arg in enumerate(args):
            op_attr = self.register_op_input(arg, node_id, port_id, op_meta)
            op_attrs.append(op_attr)

        for port_id, (name, arg) in enumerate(kwargs.items(), start=len(args)):
            op_attr = self.register_op_input(arg, node_id, port_id, op_meta)
            op_kwargs[name] = op_attr

        self.graph.add_node(
            node_id,
            type=NodeType.fn_call,
            meta=FunctionMeta(op_name=op_name, fn_name=op_meta.func.__name__, args=tuple(op_attrs), kwargs=op_kwargs),
        )

        logger.debug(f"GraphBuilderMode.process_op_inputs: {node_id=} {op_name=} {op_attrs=} {op_kwargs=}")

    def process_post_function_hooks_for_value(self, value: Any, op_meta: OpMeta, port_id: int) -> Any:
        """
        Overload process_post_function_hooks_for_value to get register output tensors to the graph

        :param output: The output of the function.
        :param op_meta: Metadata for the operation.
        :return: The modified output after post-hooks.
        """
        if isinstance(value, torch.Tensor):
            with self.disable():
                self.tensor_info[value] = TensorInfo(
                    tensor_source=TensorSource.function,
                    shape=tuple(value.shape),
                    dtype=value.dtype,
                    output_port_id=port_id,
                    source_node_id=op_meta.extra_info["node_id"],
                    name_in_model=None,
                )
        return super().process_post_function_hooks_for_value(value, op_meta, port_id)


def build_graph(model: nn.Module, *args: Any, **kwargs: Any) -> nx.MultiDiGraph:
    """
    Constructs a computational graph of the given model.

    This function builds a directed graph `nx.MultiDiGraph` representing the operations
    and data flow within the model by leveraging hooks by using GraphBuilderMode.

    :param model: The PyTorch model for which the computational graph will be built.
    :return: A nx.MultiDiGraph where nodes represent operations of model.
    """

    with torch.enable_grad():  # type: ignore
        # Gradient use to get information about __get__ functions to detect tensor.(T, mT) attributes
        with GraphBuilderMode(model=model, hook_storage=get_hook_storage(model)) as ctx:
            args, kwargs = ctx.process_model_inputs(args, kwargs)
            wrapped_forward = cast(ForwardWithHooks, model.forward)
            outputs = wrapped_forward._func(*args, **kwargs)
            outputs = ctx.process_model_outputs(outputs)
    return ctx.graph
