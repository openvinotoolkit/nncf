"""
 Copyright (c) 2019 Intel Corporation
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

import numpy as np
import torch



class TensorMeta:
    @staticmethod
    def default_comparator(lhs: 'TensorMeta', rhs: 'TensorMeta'):
        return lhs.index == rhs.index and lhs.creator_id == rhs.creator_id and lhs.shape[1:] == rhs.shape[1:]

    def __init__(self, creator_id, index, shape):
        self.creator_id = creator_id
        self.index = index
        self.shape = tuple(int(dim) for dim in shape)  # Handle cases when shape is a tuple of Tensors

    def __eq__(self, other):
        if not isinstance(other, TensorMeta):
            return False
        return self.default_comparator(self, other)

    def __hash__(self):
        return hash((self.creator_id, self.index, self.shape))

    def __str__(self):
        return "C{}_I{}_".format(self.creator_id, self.index) + "S" + "x".join([str(s) for s in self.shape])


class TracedTensor(torch.Tensor):
    # pylint: disable=abstract-method
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensor_meta = None

    @staticmethod
    def from_torch_tensor(tensor, tensor_meta: TensorMeta):
        tensor.tensor_meta = tensor_meta
        tensor.__class__ = TracedTensor
        return tensor

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


def trace_tensors(operator_output, node: 'DynamicGraphNode'):
    if isinstance(operator_output, (list, tuple)):
        output_ = []
        for i, x in enumerate(operator_output):
            meta = TensorMeta(node.node_id, i, x.shape)
            output_.append(TracedTensor.from_torch_tensor(x, meta))
        return operator_output.__class__(output_)
    if isinstance(operator_output, torch.Tensor):
        meta = TensorMeta(node.node_id, 0, operator_output.shape)
        return TracedTensor.from_torch_tensor(operator_output, meta)
    raise ValueError("Unknown return type. Can not trace function call")


def make_tensor_metas(inputs: 'OperatorInput') -> List[Optional[TensorMeta]]:
    tensor_metas = []
    for i, node_input_index_entry in enumerate(inputs):
        node_input = node_input_index_entry.getter()
        if isinstance(node_input, TracedTensor):
            tensor_metas.append(node_input.tensor_meta)
        elif isinstance(node_input, torch.Tensor) and not isinstance(node_input, TracedTensor):
            meta = TensorMeta(None, i, node_input.shape)
            tensor_metas.append(meta)
        else:
            tensor_metas.append(None)
    return tensor_metas
