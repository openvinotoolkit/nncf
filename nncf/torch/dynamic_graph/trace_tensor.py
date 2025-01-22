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
from typing import List, Optional, Tuple, Union

import torch

from nncf.common.graph.layer_attributes import Dtype


class TensorMeta:
    @staticmethod
    def default_comparator(lhs: "TensorMeta", rhs: "TensorMeta"):
        return lhs.index == rhs.index and lhs.creator_id == rhs.creator_id and lhs.shape[1:] == rhs.shape[1:]

    def __init__(
        self,
        creator_id: Union[int, None],
        index: int,
        shape: Union[List[int], Tuple[torch.Tensor, ...]],
        dtype: Dtype = Dtype.FLOAT,
    ):
        """
        :param creator_id: An ID of the node in DynamicGraph that corresponds to an operation that created the
            tensor.
        :param index: The index of this tensor in the creator operation's output.
        :param shape: The shape of the tensor.
        :param dtype: The type of the tensor
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


class TracedTensorMixin:
    """
    A mixin class providing tracing capabilities to PyTorch tensors.

    This class provides interfaces for patching a given torch tensor and associating it with the provided tensor_meta.
    """

    TENSOR_META = "tensor_meta"
    ORIGINAL_CLASS = "original_class"
    _TRACING_ATTRS = "_tracing_attrs"

    @property
    def tensor_meta(self):
        return self.tracing_attrs[TracedTensorMixin.TENSOR_META]

    @tensor_meta.setter
    def tensor_meta(self, value: Union[None, TensorMeta]):
        self.tracing_attrs[TracedTensorMixin.TENSOR_META] = value

    @property
    def tracing_attrs(self):
        if not hasattr(self, TracedTensorMixin._TRACING_ATTRS):
            self._tracing_attrs = {}
        return self._tracing_attrs

    @classmethod
    def patch(cls, tensor: torch.Tensor, tensor_meta: Optional[TensorMeta] = None) -> "TracedTensorMixin":
        """
        Patches a tensor with the TracedTensorMixin interface and associates it with the provided tensor_meta.

        :param tensor: The input tensor.
        :param tensor_meta: The metadata associated with the tensor.
        :return: The patched ternsor.
        """
        if not isinstance(tensor, TracedTensorMixin):
            original_class = tensor.__class__
            tensor.__class__ = cls
            tensor.tracing_attrs[TracedTensorMixin.ORIGINAL_CLASS] = original_class

        tensor.tensor_meta = tensor_meta
        return tensor

    def strip(self) -> None:
        """
        Reverts the tensor to its original class by removing tracing attributes.
        """
        self.__class__ = self.tracing_attrs[TracedTensorMixin.ORIGINAL_CLASS]
        delattr(self, TracedTensorMixin._TRACING_ATTRS)


class TracedTensor(torch.Tensor, TracedTensorMixin):
    """
    When tracing a torch model, intermediate tensors will be dynamically turned into
    instances of this class to be able to store additional data required for establishing
    relation between tensor producer and consumer operations.
    """

    @staticmethod
    def from_torch_tensor(tensor: torch.Tensor, tensor_meta: Optional[TensorMeta] = None) -> "TracedTensor":
        """
        Creates a TracedTensor by patching a given torch.Tensor, associating it with the provided tensor_meta.

        :param tensor: The input torch.Tensor.
        :param tensor_meta: The metadata associated with the tensor.
        :return: The resulting TracedTensor.
        """
        return TracedTensor.patch(tensor, tensor_meta)

    def as_subclass(self, cls: "TracedTensor") -> "TracedTensor":
        """
        Required for PyTorch 1.7.0 compatibility - the handle_torch_function and __torch_function__
        API in general calls this after a wrapped function call; need to preserve the tensor_meta extensions
        """

        return self

    # NOTE: This disables the __torch_function__ API altogether when using NNCF.
    # TODO: make NNCF utilize the __torch_function__ API instead.

    if hasattr(torch._C, "_disabled_torch_function_impl"):
        __torch_function__ = torch._C._disabled_torch_function_impl


class TracedParameter(torch.nn.Parameter, TracedTensorMixin):
    """
    When tracing a torch model, model parameters will be dynamically turned into
    instances of this class to be able to store additional data required for tracing parameters
    across operations.
    """

    NAME = "name"
    IS_REUSED = "is_reused"

    @property
    def name(self):
        return self.tracing_attrs[TracedParameter.NAME]

    @property
    def is_reused(self):
        return self.tracing_attrs[TracedParameter.IS_REUSED]

    def get_dtype(self):
        # Type of self is TracedParameter or TracedTensor
        return super(self.__class__, self).__getattribute__("dtype")

    def __getattribute__(self, name):
        if name == "dtype":
            return self.get_dtype()
        return super().__getattribute__(name)

    @staticmethod
    def from_torch_parameter(tensor: torch.nn.Parameter, name: str, is_reused: bool) -> "TracedParameter":
        """
        Creates a TracedParameter by patching a given torch.nn.Parameter, associating it
        with the provided parameter name.

        :param tensor: The input torch.nn.Parameter.
        :param name: The parameter name.
        :param is_reused: True if parameter is used as an input in several operations of the model otherwise False.
        :return: The resulting TracedParameter.
        """
        TracedParameter.patch(tensor)
        tensor.tracing_attrs[TracedParameter.NAME] = name
        tensor.tracing_attrs[TracedParameter.IS_REUSED] = is_reused
        return tensor

    def as_subclass(self, cls: "TracedParameter") -> "TracedParameter":
        """
        Required for PyTorch 1.7.0 compatibility - the handle_torch_function and __torch_function__
        API in general calls this after a wrapped function call; need to preserve the tensor_meta extensions
        """

        return self

    # NOTE: This disables the __torch_function__ API altogether when using NNCF.
    # TODO: make NNCF utilize the __torch_function__ API instead.

    if hasattr(torch._C, "_disabled_torch_function_impl"):
        __torch_function__ = torch._C._disabled_torch_function_impl
