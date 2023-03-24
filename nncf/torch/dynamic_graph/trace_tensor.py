"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
import torch

from nncf import nncf_logger
from nncf.common.graph import Dtype


class TensorMeta:
    @staticmethod
    def default_comparator(lhs: 'TensorMeta', rhs: 'TensorMeta'):
        return lhs.index == rhs.index and lhs.creator_id == rhs.creator_id and lhs.shape[1:] == rhs.shape[1:]

    def __init__(self, creator_id: int, index: int, shape: Union[List[int], Tuple[torch.Tensor, ...]],
                 dtype: Dtype = Dtype.FLOAT):
        """
        :param creator_id: An ID of the node in DynamicGraph that corresponds to an operation that created the
            tensor.
        :param index: The index of this tensor in the creator operation's output.
        :param shape: The shape of the tensor.
        """
        self.creator_id = creator_id
        self.index = index
        self.shape = tuple(int(dim) for dim in shape)  # Handle cases when shape is a tuple of Tensors
        self.dtype = dtype

    def __eq__(self, other):
        if not isinstance(other, TensorMeta):
            return False
        return self.default_comparator(self, other)

    def __hash__(self):
        return hash((self.creator_id, self.index, self.shape))

    def __str__(self):
        return "C{}_I{}_".format(self.creator_id, self.index) + "S" + "x".join([str(s) for s in self.shape])


class TracedTensor(torch.Tensor):
    """
    When tracing a torch model, intermediate tensors will be dynamically turned into
    instances of this class to be able to store additional data required for establishing
    relation between tensor producer and consumer operations.
    """
    @staticmethod
    def from_torch_tensor(tensor, tensor_meta: TensorMeta):
        tensor.tensor_meta = tensor_meta
        tensor.__class__ = TracedTensor
        #pylint:disable=protected-access
        tensor._nncf_expired = False
        return tensor

    def nncf_expire(self):
        """
        Mark the traced tensor as "expired". The tensor's metainformation should
        then be considered outdated/invalid.
        """
        self._nncf_expired = True

    @property
    def nncf_expired(self) -> bool:
        return self._nncf_expired

    def as_subclass(self, cls: 'TracedTensor') -> 'TracedTensor':
        """
        Required for PyTorch 1.7.0 compatibility - the handle_torch_function and __torch_function__
        API in general calls this after a wrapped function call; need to preserve the tensor_meta extensions
        """

        return self

    # NOTE: This disables the __torch_function__ API altogether when using NNCF.
    # TODO: make NNCF utilize the __torch_function__ API instead.
    #pylint:disable=protected-access
    if hasattr(torch._C, "_disabled_torch_function_impl"):
        __torch_function__ = torch._C._disabled_torch_function_impl


def is_iterable(item):
    non_iterable_types = (str, bytes, bytearray, torch.Tensor, np.ndarray)
    # pylint:disable=isinstance-second-argument-not-valid-type
    return isinstance(item, Iterable) and not isinstance(item, non_iterable_types)


def flatten(items):
    it = items.items() if hasattr(items, 'items') else iter(items)
    for item in it:
        if is_iterable(item):
            for i in flatten(item):
                yield i
        else:
            yield item


def flatten_args(args, kwargs):
    return list(flatten(args)) + list(flatten(kwargs))


def get_dtype(x: torch.Tensor) -> Dtype:
    if x.dtype in [torch.float, torch.float16, torch.float32, torch.float64]:
        return Dtype.FLOAT
    return Dtype.INTEGER


TensorOrTupleOrList = TypeVar('TensorOrTupleOrList', List[torch.Tensor], Tuple[torch.Tensor], torch.Tensor)


def trace_tensors(operator_output: TensorOrTupleOrList,
                  node: 'DynamicGraphNode', ctx: 'TracingContext' = None) -> TensorOrTupleOrList:
    """
    Dynamically turn torch.Tensor instances in `operator_output` into TracedTensor instances. `operator_output` is
    presumed to be the output of a model operation (function call) associated with `node`.
    :param operator_output: The output of an NNCF-wrapped function executed in a model object.
    :param node: A node in DynamicGraph associated with the function that produced `operator_output`
    :param ctx: If supplied, the resulting tensors will be registered within this TracingContext instance
    to be marked as expired on context exit, which is required to correctly process situations when a traced model
    retains intermediate tensor values.
    :return: Same structure as `operator_output`, but with torch.Tensor entries turned into TracedTensors.
    """
    if isinstance(operator_output, (list, tuple)):
        output_ = []
        for i, x in enumerate(operator_output):
            meta = None
            if node is not None:
                meta = TensorMeta(node.node_id, i, x.shape, get_dtype(x))
            tt = TracedTensor.from_torch_tensor(x, meta)
            if ctx is not None:
                ctx.register_traced_tensor(tt)
            output_.append(tt)
        return operator_output.__class__(output_)
    if isinstance(operator_output, torch.Tensor):
        meta = None
        if node is not None:
            meta = TensorMeta(node.node_id, 0, operator_output.shape, get_dtype(operator_output))
        tt = TracedTensor.from_torch_tensor(operator_output, meta)
        if ctx is not None:
            ctx.register_traced_tensor(tt)
        return tt
    nncf_logger.debug(f"Could not find tensors to trace in operator output: {operator_output}")
    return operator_output


def make_tensor_metas(inputs: 'OperatorInput') -> List[Optional[TensorMeta]]:
    """
    Produces TensorMeta data for each torch.Tensor or TracedTensor in `inputs`.
    :param inputs: An OperatorInput representation of input arguments to an operation in the traced model.
    :return: A list of TensorMeta objects, one for every torch.Tensor or TracedTensor object in `inputs` in the
    order of item enumeration in `inputs`.
    """
    tensor_metas = []
    for i, node_input_index_entry in enumerate(inputs):
        node_input = node_input_index_entry.getter()
        if isinstance(node_input, TracedTensor):
            if not node_input.nncf_expired:
                tensor_metas.append(node_input.tensor_meta)
            else:
                meta = TensorMeta(None, i, node_input.shape)
                tensor_metas.append(meta)
        elif isinstance(node_input, torch.Tensor) and not isinstance(node_input, TracedTensor):
            meta = TensorMeta(None, i, node_input.shape)
            tensor_metas.append(meta)
        else:
            tensor_metas.append(None)
    return tensor_metas
