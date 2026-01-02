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
import pytest
import torch

from nncf.torch.return_types import _TORCH_RETURN_TYPES
from nncf.torch.return_types import maybe_get_values_from_torch_return_type
from nncf.torch.return_types import maybe_wrap_to_torch_return_type


@pytest.mark.parametrize("return_type", _TORCH_RETURN_TYPES)
def test_unwrap_wrap_torch_return_type(return_type):
    wrapped_tensor = return_type((torch.tensor(0), torch.tensor(1)))
    assert wrapped_tensor.values == torch.tensor(0)
    unwrapped_tensor = maybe_get_values_from_torch_return_type(wrapped_tensor)
    assert unwrapped_tensor == torch.tensor(0)

    updated_wrapped_tensor = maybe_wrap_to_torch_return_type(unwrapped_tensor, wrapped_tensor)
    assert updated_wrapped_tensor == wrapped_tensor


@pytest.mark.parametrize(
    "input_", [torch.tensor(0), [torch.tensor(0), torch.tensor(1)], (torch.tensor(0), torch.tensor(1))]
)
def test_wrap_unwrap_do_nothing_to_tensor(input_):
    wrapped_input = maybe_get_values_from_torch_return_type(input_)
    assert wrapped_input is input_
    unwrapped_input = maybe_wrap_to_torch_return_type(input_, wrapped_input)
    assert unwrapped_input is input_
