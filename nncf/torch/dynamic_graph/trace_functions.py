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

from copy import deepcopy
from typing import Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

import numpy as np
import torch

import nncf
from nncf import nncf_logger
from nncf.common.graph.layer_attributes import Dtype
from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.dynamic_graph.graph import DynamicGraphNode
from nncf.torch.dynamic_graph.op_input_processing import OperatorInput
from nncf.torch.dynamic_graph.trace_tensor import TensorMeta
from nncf.torch.dynamic_graph.trace_tensor import TracedTensor
from nncf.torch.dynamic_graph.trace_tensor import TracedTensorMixin
from nncf.torch.nested_objects_traversal import objwalk

TensorOrTupleOrList = TypeVar("TensorOrTupleOrList", List[torch.Tensor], Tuple[torch.Tensor], torch.Tensor)


def is_iterable(item):
    non_iterable_types = (str, bytes, bytearray, torch.Tensor, np.ndarray)

    return isinstance(item, Iterable) and not isinstance(item, non_iterable_types)


def flatten(items):
    it = items.items() if hasattr(items, "items") else iter(items)
    for item in it:
        if is_iterable(item):
            for i in flatten(item):
                yield i
        else:
            yield item


def flatten_args(args, kwargs):
    return list(flatten(args)) + list(flatten(kwargs))


def get_dtype(x: torch.Tensor) -> Dtype:
    if x.dtype in [torch.float, torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        return Dtype.FLOAT
    return Dtype.INTEGER


def forward_trace_only(operator: Callable, *args, **kwargs):
    """
    This wrapper override will result in the operator not being added to graph,
    but the result will still have TracedTensors with parent IDs left the same as in input.
    Useful for operators which are not likely to be present in patterns considered for
    compression, but still have to be accounted for so that the NNCF internal graph representation
    does not become disjoint.
    """

    result = operator(*args, **kwargs)

    fargs = flatten_args(args, kwargs)
    input_traced_tensor_indices = [i for i in range(len(fargs)) if isinstance(fargs[i], TracedTensor)]

    if isinstance(result, (list, tuple)):
        output_tensors_to_be_traced_indices = [i for i in range(len(result)) if isinstance(result[i], torch.Tensor)]

        was_tuple = isinstance(result, tuple)
        result = list(result)
        if len(input_traced_tensor_indices) == 1:
            # Broadcast one and the same creator ID of input to all outputs
            for out_idx in output_tensors_to_be_traced_indices:
                forwarded_meta = deepcopy(fargs[input_traced_tensor_indices[0]].tensor_meta)
                if forwarded_meta is not None:
                    forwarded_meta.shape = tuple(result[out_idx].shape)
                result[out_idx] = TracedTensor.from_torch_tensor(result[out_idx], forwarded_meta)
        elif len(input_traced_tensor_indices) != len(output_tensors_to_be_traced_indices):
            raise nncf.ValidationError(
                "Unable to forward trace through operator {} - "
                "input and output tensor count mismatch!".format(operator.__name__)
            )
        else:
            # Assume that output tensor order corresponds to input tensor order
            for in_idx, out_idx in zip(input_traced_tensor_indices, output_tensors_to_be_traced_indices):
                forwarded_meta = deepcopy(fargs[in_idx].tensor_meta)
                if forwarded_meta is not None:
                    forwarded_meta.shape = tuple(result[out_idx].shape)
                result[out_idx] = TracedTensor.from_torch_tensor(result[out_idx], forwarded_meta)
        if was_tuple:
            result = tuple(result)
    elif len(input_traced_tensor_indices) > 1:
        raise nncf.ValidationError(
            "Unable to forward trace through operator {} - "
            "input and output tensor count mismatch!".format(operator.__name__)
        )
    elif input_traced_tensor_indices:
        forwarded_meta = deepcopy(fargs[input_traced_tensor_indices[0]].tensor_meta)
        if forwarded_meta is not None:
            forwarded_meta.shape = tuple(result.shape)
        return TracedTensor.from_torch_tensor(result, forwarded_meta)
    # No traced tensors in input, return a usual torch.Tensor as well
    return result


def trace_tensor(x: torch.Tensor, port_id: int, node: DynamicGraphNode, ctx: TracingContext) -> torch.Tensor:
    """
    Dynamically turn torch.Tensor instance into tensor with tracing capabilities instance using TracedTensorMixin.

    :param x: The input Tensor.
    :param node: A node in DynamicGraph associated with the function that produced the input tensor
    :param ctx: The resulting tensors will be registered within this TracingContext instance
        to be stipped on context exit, which is required to correctly process situations when a traced model
        retains intermediate tensor values.
    :return: The resulting tensor with the TracedTensorMixin interface.
    """
    if not isinstance(x, torch.Tensor):
        nncf_logger.debug(f"Could not find tensors to trace in operator output: {x}")
        return x

    meta = None
    if node is not None:
        meta = TensorMeta(node.node_id, port_id, x.shape, get_dtype(x))

    if isinstance(x, TracedTensorMixin):
        tt = x
        tt.tensor_meta = meta
    else:
        tt = TracedTensor.from_torch_tensor(x, meta)
    ctx.register_traced_tensor(tt)
    return tt


def trace_tensors(
    operator_output: TensorOrTupleOrList, node: "DynamicGraphNode", ctx: "TracingContext"
) -> TensorOrTupleOrList:
    """
    Dynamically turn torch.Tensor instances in `operator_output` into tensor with tracing capabilities instances
    using TracedTensorMixin. `operator_output` is presumed to be the output of a model operation (function call)
    associated with `node`.

    :param operator_output: The output of an NNCF-wrapped function executed in a model object.
    :param node: A node in DynamicGraph associated with the function that produced `operator_output`
    :param ctx: The resulting tensors will be registered within this TracingContext instance
        to be stipped on context exit, which is required to correctly process situations when a traced model
        retains intermediate tensor values.
    :return: Same structure as `operator_output`, but with torch.Tensor entries turned into tensor
        with tracing capabilities instance using TracedTensorMixin.
    """

    if isinstance(operator_output, (list, tuple)):
        output_ = []
        for i, x in enumerate(operator_output):
            tt = trace_tensor(x, i, node, ctx)
            output_.append(tt)
        return operator_output.__class__(output_)
    return trace_tensor(operator_output, 0, node, ctx)


def make_tensor_metas(inputs: OperatorInput) -> List[Optional[TensorMeta]]:
    """
    Produces TensorMeta data for each torch.Tensor or TracedTensorMixin in `inputs`.

    :param inputs: An OperatorInput representation of input arguments to an operation in the traced model.
    :return: A list of TensorMeta objects, one for every torch.Tensor or TracedTensorMixin object in `inputs` in the
    order of item enumeration in `inputs`.
    """
    tensor_metas = []
    for i, node_input_index_entry in enumerate(inputs):
        node_input = node_input_index_entry.getter()
        if isinstance(node_input, TracedTensorMixin):
            tensor_metas.append(node_input.tensor_meta)
        elif isinstance(node_input, torch.Tensor):
            meta = TensorMeta(None, i, node_input.shape, get_dtype(node_input))
            tensor_metas.append(meta)
        else:
            tensor_metas.append(None)
    return tensor_metas


def strip_traced_tensors(args: Tuple, kwargs: Dict) -> Tuple[Tuple, Dict]:
    """
    Required to guard against new forward calls on tensors that have already passed
    through NNCF's forward once and got turned into TracedTensors by reference access.
    """
    is_traced_tensor_predicate = lambda x: isinstance(x, TracedTensorMixin)
    strip_traced_tensor = lambda x: x.strip()

    args = objwalk(args, is_traced_tensor_predicate, strip_traced_tensor)
    kwargs = objwalk(kwargs, is_traced_tensor_predicate, strip_traced_tensor)
    return args, kwargs
