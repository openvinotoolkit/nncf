# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional, Tuple, Type, Union

import torch


def __get_supported_torch_return_types() -> Tuple[Type[tuple], ...]:
    """
    Collects types from torch.return_type which can be wrapped/unwrapped by NNCF.
    NNCF can wrap/unwrap only public return types that have `values` attribute.

    :return: List of types from torch.return_type which can be wrapped/unwrapped by NNCF.
    """
    return_type_names = [t for t in dir(torch.return_types) if not t.startswith("_")]
    return_types = [getattr(torch.return_types, t_name) for t_name in return_type_names]
    return_types = [t for t in return_types if hasattr(t, "values")]
    return tuple(return_types)


_TORCH_RETURN_TYPES = __get_supported_torch_return_types()


def maybe_unwrap_from_torch_return_type(tensor: Any) -> torch.Tensor:
    """
    Attempts to unwrap the tensor value from one of torch.return_types instances
    in case torch operation output is wrapped by a torch return_type.

    :param tensor: Torch tensor or torch return type instance to unwrap values from.
    :return: Unwrapped torch tensor.
    """
    if isinstance(tensor, _TORCH_RETURN_TYPES):
        return tensor.values
    return tensor


def maybe_wrap_to_torch_return_type(tensor: torch.Tensor, wrapped_input: Optional[Union[tuple, torch.Tensor]]) -> Any:
    """
    Wraps tensor to wrapped_input wrapper in case wrapped_input is wrapped by a torch.return_value container.

    :param tensor: Torch tensor to wrap.
    :param wrapped_tensor: Instance of the tensor before it was unwrapped.
    :return: Wrapped tensor in case wrapped_input is wrapped by a torch.return_value container else the tensor.
    """

    if isinstance(wrapped_input, _TORCH_RETURN_TYPES):
        return wrapped_input.__class__([tensor, *wrapped_input[1:]])
    return tensor
